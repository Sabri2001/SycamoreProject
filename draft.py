import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

def main():
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    fig1, ax1 = plt.subplots()
    ax1.plot(x, y1, label='Sinusoidal Plot')
    ax1.set_title('Figure 1: Sinusoidal Plot')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.legend()
    ax1.grid(True)

    fig2, ax2 = plt.subplots()
    ax2.plot(x, y2, label='Cosine Plot', linestyle='--')
    ax2.set_title('Figure 2: Cosine Plot')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    ax2.legend()
    ax2.grid(True)

    # fig3, ax3 = plt.subplots()
    # ax3.plot(x, y1, label='Sinusoidal Plot')
    # ax3.set_title('Figure 1: Sinusoidal Plot')
    # ax3.set_xlabel('X-axis')
    # ax3.set_ylabel('Y-axis')
    # ax3.legend()
    # ax3.grid(True)

    # Create the Tkinter window
    window = tk.Tk()
    window.title("Matplotlib Animation Window")

    # Create a button
    button = tk.Button(window, text="Close Figures", command=lambda: on_button_press(fig1, fig2, window))
    button.pack(pady=10)

    # Embed Matplotlib figures in Tkinter window
    canvas1 = FigureCanvasTkAgg(fig1, master=window)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    canvas2 = FigureCanvasTkAgg(fig2, master=window)
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

    window.mainloop()

    print("WINDOW CLOSED")

def on_button_press(fig1, fig2, window):
    # Close Matplotlib figures
    plt.close(fig1)
    plt.close(fig2)

    # Close the Tkinter window
    window.destroy()
    window.quit()

    
if __name__ == "__main__":
    main()
