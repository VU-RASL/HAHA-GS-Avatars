import cv2

# Step 1: Load the given UV map , 512x512 texture
input_texture_path = "texture.png"
uv_map = cv2.imread(input_texture_path)

# Step 2: Resize to 4096x4096
resized_uv_map = cv2.resize(uv_map, (4096, 4096), interpolation=cv2.INTER_LINEAR)

# Step 3: Rotate clockwise by 180 degrees
rotated_uv_map = cv2.rotate(resized_uv_map, cv2.ROTATE_180)

# Step 4: Flip left and right
flipped_uv_map = cv2.flip(rotated_uv_map, 1)

# Save the final result
output_texture_path = "path_to_output_texture.png"
cv2.imwrite(output_texture_path, flipped_uv_map)

print("UV map successfully transformed and saved to:", output_texture_path)