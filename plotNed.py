import re
import matplotlib.pyplot as plt
with open('your_file.txt', 'r') as file:
    data = file.read()

# Step 2: Extract x_m, y_m, and z_m values using regex
pattern = r'x_m:\s*(-?\d+\.\d+e?-?\d*),\s*y_m:\s*(-?\d+\.\d+e?-?\d*),\s*z_m:\s*(-?\d+\.\d+e?-?\d*)'
matches = re.findall(pattern, data)

# Convert extracted values to float
x_vals = [float(match[0]) for match in matches]
y_vals = [float(match[1]) for match in matches]
z_vals = [float(match[2]) for match in matches]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_vals, y_vals, z_vals, c='b', marker='o')

ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Z Position (m)')
ax.set_title('3D Scatter Plot of Position')

plt.show()
