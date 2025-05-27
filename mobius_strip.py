"""importing libraries required for this task"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad, quad

""" defining class name with Mobius Strip """
class MobiusStrip:
    def __init__(self, R=1, w=1, n=100):
        """Initialing radius R, width w, and resolution n"""
        self.R = R
        self.w = w
        self.n = n

        # Create grid of u and v values
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)
        self.u, self.v = np.meshgrid(self.u, self.v)

        # Compute x, y, z coordinates using parametric equations
        self.x, self.y, self.z = self.compute_surface()

    def compute_surface(self):
        """Computing the (x, y, z) points on the Möbius surface"""
        u, v, R = self.u, self.v, self.R
        x = (R + v * np.cos(u / 2)) * np.cos(u)
        y = (R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        return x, y, z

    def surface_area(self):
        """Approximate the surface area using double integration"""
        def integrand(v, u):
            # Partial derivatives
            dxdu = np.array([
                -(self.R + v * np.cos(u / 2)) * np.sin(u) - v/2 * np.sin(u / 2) * np.cos(u),
                (self.R + v * np.cos(u / 2)) * np.cos(u) - v/2 * np.sin(u / 2) * np.sin(u),
                v/2 * np.cos(u / 2)
            ])
            dxdv = np.array([
                np.cos(u / 2) * np.cos(u),
                np.cos(u / 2) * np.sin(u),
                np.sin(u / 2)
            ])
            # Area element = magnitude of cross product
            return np.linalg.norm(np.cross(dxdu, dxdv))

        area, _ = dblquad(integrand, 0, 2 * np.pi, 
                          lambda u: -self.w / 2, 
                          lambda u: self.w / 2)
        return area

    def edge_length(self):
        """Approximate the length of the boundary edge (loop around strip)"""
        def integrand(u):
            dxdu = np.array([
                -(self.R + self.w/2 * np.cos(u / 2)) * np.sin(u) - (self.w/4) * np.sin(u / 2) * np.cos(u),
                (self.R + self.w/2 * np.cos(u / 2)) * np.cos(u) - (self.w/4) * np.sin(u / 2) * np.sin(u),
                (self.w/4) * np.cos(u / 2)
            ])
            return np.linalg.norm(dxdu)

        length, _ = quad(integrand, 0, 2 * np.pi)
        return length

    def plot(self):
        """Visualize the Möbius strip in 3D"""
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.x, self.y, self.z, cmap='viridis', edgecolor='k', alpha=0.9)
        ax.set_title("Möbius Strip")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()

# ---------------- MAIN SCRIPT ----------------
if __name__ == "__main__":
    # Create a Möbius strip instance
    strip = MobiusStrip(R=5, w=5, n=100)

    # Plot the strip
    strip.plot()

    # Display surface area and edge length
    print("Surface Area ≈", round(strip.surface_area(), 4))
    print("Edge Length ≈", round(strip.edge_length(), 4))
