import matplotlib.pyplot as plt

def plot_curves(out, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(12,4))

    # cumulative reach
    axes[0].plot(out["ever_seen_F"], label="Fake — ever seen")
    axes[0].plot(out["ever_seen_R"], label="Real — ever seen")
    axes[0].set_title(f"Cumulative Reach — {title}")
    axes[0].set_xlabel("Time"); axes[0].set_ylabel("Users"); axes[0].legend()

    # posters per step
    axes[1].plot(out["posters_F"], label="Fake — posters")
    axes[1].plot(out["posters_R"], label="Real — posters")
    axes[1].set_title(f"Posters per Step — {title}")
    axes[1].set_xlabel("Time"); axes[1].set_ylabel("Users posting"); axes[1].legend()

    plt.tight_layout(); plt.show()
