# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 19:02:12 2023

@author: mgt
"""

import tkinter as tk


import sys
from ttkwidgets import TickScale

from tkvideoutils import VideoPlayer
from tkinter import filedialog, messagebox


def on_closing():
    player.loading = False
    root.quit()
    root.destroy()


if __name__ == '__main__':
    # create instance of window
    root = tk.Tk()
    # set window title
    root.title('Video Player')
    # load images
    pause_image = tk.PhotoImage(file="C:\\Users\\haich\\OneDrive\\Desktop\\ElderMonitoring\\icon\\video-pause-button.png")
    play_image = tk.PhotoImage(file="C:\\Users\\haich\\OneDrive\\Desktop\\ElderMonitoring\\icon\\play-button.png")
    skip_backward = tk.PhotoImage(file="C:\\Users\\haich\\OneDrive\\Desktop\\ElderMonitoring\\icon\\fast-backward.png")
    skip_forward = tk.PhotoImage(file="C:\\Users\\haich\\OneDrive\\Desktop\\ElderMonitoring\\icon\\fast-forward.png")
    # create user interface
    button = tk.Button(root, image=play_image)
    forward_button = tk.Button(root, image=skip_forward)
    backward_button = tk.Button(root, image=skip_backward)
    video_label = tk.Label(root)
    video_path = r"WIN_20230330_19_03_20_Pro.mp4"
    audio_path = "C:\\Users\\haich\\OneDrive\\Desktop\\ElderMonitoring\\SoundEventClassifier\\VideoAudioSave.wav"
    slider_var = tk.IntVar(root)
    slider = TickScale(root, orient="horizontal", variable=slider_var)
    # place elements
    video_label.pack()
    button.pack()
    forward_button.pack()
    backward_button.pack()
    slider.pack()
    loading_gif = "C:\\Users\\haich\\OneDrive\\Desktop\\ElderMonitoring\\puzzle.png"
    if video_path:
        # read video to display on label
        player = VideoPlayer(root, video_path, audio_path, video_label,loading_gif, size=(700, 500),
                             play_button=button, play_image=play_image, pause_image=pause_image,
                             slider=slider, slider_var=slider_var, keep_ratio=True, cleanup_audio=True)
    else:
        messagebox.showwarning("Select Video File", "Please retry and select a video file.")
        sys.exit(1)
    player.set_clip(50, 70)
    forward_button.config(command=player.skip_video_forward)
    backward_button.config(command=player.skip_video_backward)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()