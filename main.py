"""Entry point mirroring the original test script using the refactored modules."""

from visibility_animation import PhobosVisibilityAnimation

if __name__ == "__main__":
    # PhobosVisibilityAnimation().save()
    # Open the file in the default viewer
    import os
    os.system(f"open {PhobosVisibilityAnimation().save()}")
