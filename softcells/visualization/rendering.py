import numpy as np
import skimage as ski

def create_masks(single_df, SCREEN_HEIGHT, SCREEN_WIDTH, subsample_points, screen_downsample):
    
    SCREEN_WIDTH = int(SCREEN_WIDTH / screen_downsample)
    SCREEN_HEIGHT = int(SCREEN_HEIGHT / screen_downsample)
    shifts = [
        (0, 0), (SCREEN_WIDTH, 0), (-SCREEN_WIDTH, 0),
        (0, SCREEN_HEIGHT), (0, -SCREEN_HEIGHT),
        (SCREEN_WIDTH, SCREEN_HEIGHT), (-SCREEN_WIDTH, SCREEN_HEIGHT),
        (SCREEN_WIDTH, -SCREEN_HEIGHT), (-SCREEN_WIDTH, -SCREEN_HEIGHT),
    ]

    
    all_ids = single_df['cell_unique_id'].unique()
    masks_cyto = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=int)

    for cell_id in all_ids:
        polygon_df = single_df.query(f"cell_unique_id == {cell_id} & identity == 0")

        if polygon_df.empty:
            continue

        x_coords = polygon_df['x_world'].values[::subsample_points] /screen_downsample
        y_coords = polygon_df['y_world'].values[::subsample_points] /screen_downsample
                
        # 1. Calculate the difference between consecutive points
        # Using np.diff tells us if there was a large jump
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)

        # 2. Identify where wrapping occurs
        # A jump > half the screen width indicates a wrap.
        # We add a 0 at the start to make the array the same size as the vertices.
        x_wraps = np.concatenate(([0], dx < -SCREEN_WIDTH * 0.5)) - np.concatenate(([0], dx > SCREEN_WIDTH * 0.5))
        y_wraps = np.concatenate(([0], dy < -SCREEN_HEIGHT * 0.5)) - np.concatenate(([0], dy > SCREEN_HEIGHT * 0.5))
        
        # 3. Propagate the winding using cumulative sum
        # This ensures that once a wrap occurs, all following points get the new winding number.
        winding_x_corrected = np.cumsum(x_wraps)
        winding_y_corrected = np.cumsum(y_wraps)
        
        # "Unwrap" the coordinates using our newly calculated winding numbers
        x_unwrapped = x_coords + winding_x_corrected * SCREEN_WIDTH
        y_unwrapped = y_coords + winding_y_corrected * SCREEN_HEIGHT
        
        # Draw the complete polygon and its 8 ghosts
        for dx_shift, dy_shift in shifts:
            shifted_x = x_unwrapped + dx_shift
            shifted_y = y_unwrapped + dy_shift
            
            rr, cc = ski.draw.polygon(shifted_y, shifted_x, shape=masks_cyto.shape)
            masks_cyto[rr, cc] = cell_id + 1


    masks_nuc = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=int)
    for cell_id in all_ids:
        polygon_df = single_df.query(f"cell_unique_id == {cell_id} & identity == 1")

        if polygon_df.empty:
            continue

        x_coords = polygon_df['x_world'].values[::subsample_points] /screen_downsample
        y_coords = polygon_df['y_world'].values[::subsample_points] /screen_downsample
        
        # --- START: Winding Recalculation ---
        
        # 1. Calculate the difference between consecutive points
        # Using np.diff tells us if there was a large jump
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)

        # 2. Identify where wrapping occurs
        # A jump > half the screen width indicates a wrap.
        # We add a 0 at the start to make the array the same size as the vertices.
        x_wraps = np.concatenate(([0], dx < -SCREEN_WIDTH * 0.5)) - np.concatenate(([0], dx > SCREEN_WIDTH * 0.5))
        y_wraps = np.concatenate(([0], dy < -SCREEN_HEIGHT * 0.5)) - np.concatenate(([0], dy > SCREEN_HEIGHT * 0.5))
        
        # 3. Propagate the winding using cumulative sum
        # This ensures that once a wrap occurs, all following points get the new winding number.
        winding_x_corrected = np.cumsum(x_wraps)
        winding_y_corrected = np.cumsum(y_wraps)

        # --- END: Winding Recalculation ---
        
        # "Unwrap" the coordinates using our newly calculated winding numbers
        x_unwrapped = x_coords + winding_x_corrected * SCREEN_WIDTH
        y_unwrapped = y_coords + winding_y_corrected * SCREEN_HEIGHT
        
        # Draw the complete polygon and its 8 ghosts
        for dx_shift, dy_shift in shifts:
            shifted_x = x_unwrapped + dx_shift
            shifted_y = y_unwrapped + dy_shift
            
            rr, cc = ski.draw.polygon(shifted_y, shifted_x, shape=masks_nuc.shape)
            masks_nuc[rr, cc] = cell_id + 1

    return masks_cyto, masks_nuc

import noise
def nuc_perlin(shape):
    # Perlin noise for nuclei
    scale = 100.0
    octaves = 3
    persistence = 2
    lacunarity = 2.0
    base = 0
    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            world[i][j] = noise.pnoise2(i / scale, 
                                        j / scale, 
                                        octaves=octaves, 
                                        persistence=persistence, 
                                        lacunarity=lacunarity, 
                                        repeatx=shape[0], 
                                        repeaty=shape[1], 
                                        base=base)
    world = world - world.min()
    return world

def cyto_perlin(shape):
    scale = 100.0
    octaves = 4
    persistence = 3
    lacunarity = 2.0
    base = 0
    world2 = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            world2[i][j] = noise.pnoise2((i) / scale, 
                                        (j) / scale, 
                                        octaves=octaves, 
                                        persistence=persistence, 
                                        lacunarity=lacunarity, 
                                        repeatx=shape[0], 
                                        repeaty=shape[1], 
                                        base=base)
    world2 = (world2 - world2.min()) / (world2.max() - world2.min())
    world2 = (world2 + 1) / 2
    return world2