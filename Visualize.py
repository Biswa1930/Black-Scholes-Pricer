import numpy as np
import matplotlib.pyplot as plt
from BS import BlackScholesEngine

# 1. Define the parameter ranges for the market
spot_prices = np.linspace(80, 120, 50)        # Stock prices from $80 to $120
times_to_maturity = np.linspace(0.01, 1.0, 50) # Time from 0.01 years (approx 3 days) to 1 year

# 2. Create a 2D meshgrid for our 3D plot
S_grid, T_grid = np.meshgrid(spot_prices, times_to_maturity)

# 3. Initialize the engine with arrays instead of single numbers
# We are holding Strike (100), Rate (5%), and Volatility (20%) constant
engine = BlackScholesEngine(S=S_grid, K=100, T=T_grid, r=0.05, sigma=0.20)

# 4. Calculate Gamma for the entire surface instantly
gamma_surface = engine.gamma()

# 5. Build the 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface using a colormap
surf = ax.plot_surface(S_grid, T_grid, gamma_surface, cmap='viridis', edgecolor='none')

# Add professional labels
ax.set_title('Black-Scholes Gamma Surface\n(Strike=100, Vol=20%, r=5%)')
ax.set_xlabel('Underlying Asset Price ($)')
ax.set_ylabel('Time to Maturity (Years)')
ax.set_zlabel('Gamma ($\Gamma$)')
fig.colorbar(surf, shrink=0.5, aspect=5, label='Gamma Value')

# 6. Save the image directly to the folder for your GitHub README
plt.savefig('gamma_surface.png', dpi=300, bbox_inches='tight')
print("Success! 'gamma_surface.png' has been saved to your project folder.")

# 7. Display the interactive plot on your screen
plt.show()