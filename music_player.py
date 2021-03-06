from pygame import mixer
from tkinter import Tk
from tkinter import Label
from tkinter import filedialog
from tkinter import Button


current_volume = float(0.5)

# Functions        

def play_song():
    filename = filedialog.askopenfilename(initialdir="C:/" , title="Select a file")
    current_song = filename
    song_title = filename.split("/")
    song_title = song_title[-1]
    print(song_title)

    try:
        mixer.init()
        mixer.music.load(current_song)
        mixer.music.set_volume(current_volume)
        mixer.music.play()
        song_title_label.config(fg='green', text="Now Playing : " + str(song_title))
        volume_label.config(fg='green', text="Volume : " + str(current_volume))

    except Exception as e:
        print(e)
        song_title_label.config(fg="red", text="Error Playing track")

def reduce_volume():
    try:
        global current_volume
        if current_volume <= 0:
            volume_label.config(fg="red", text="Volume Muted")
            return
        current_volume = current_volume - float(0.1)
        current_volume = round(current_volume,1)
        mixer.music.set_volume(current_volume)
        volume_label.config(fg="green", text="Volume : " + str(current_volume))

    except Exception as e:
        print(e)
        song_title_label.config(fg="red", text="Track hasn't been selected yet")


def increase_volume():
    try:
        global current_volume
        if current_volume >= 1:
            volume_label.config(fg="green", text="Volume Full")
            return
        current_volume = current_volume + float(0.1)
        current_volume = round(current_volume,1)
        mixer.music.set_volume(current_volume)
        volume_label.config(fg="green", text="Volume : " + str(current_volume))

    except Exception as e:
        print(e)
        song_title_label.config(fg="red", text="Track hasn't been selected yet")




def pause():
    try:
        mixer.music.pause()
    
    except Exception as e:
        print(e)
        song_title_label.config(fg="red", text="Track hasn't been selected yet")

def resume():
    try:
        mixer.music.unpause()
    
    except Exception as e:
        print(e)
        song_title_label.config(fg="red", text="Track hasn't been selected yet")

# Main Screen
master = Tk()
master.title("Music Player")

# Labels
Label(master, text="Custom Music Player", 
font=("Calibri",15),fg='red').grid(sticky="N", row=0, padx=120)

Label(master, text="Select a track", 
font=("Calibri",12),fg='blue').grid(sticky="N", row=1)

Label(master, text="Volume", 
font=("Calibri",12),fg='red').grid(sticky="N", row=4)

song_title_label = Label(master , font=("Calibri,12"))
song_title_label.grid(sticky='N', row=3)

volume_label = Label(master , font=("Calibri,12"))
volume_label.grid(sticky='N', row=5)

# Button
Button(master, text="Select track", 
font=("Calibri",12),fg='black', command=play_song).grid(sticky="N", row=2)

Button(master, text="Pause", 
font=("Calibri",12),fg='black', command=pause).grid(sticky="E", row=3)

Button(master, text="Resume", 
font=("Calibri",12),fg='black', command=resume).grid(sticky="W", row=3)

Button(master, text="-", 
font=("Calibri",12),fg='black', width=5, command=reduce_volume).grid(sticky="W", row=5)

Button(master, text="+", 
font=("Calibri",12),fg='black', width=5, command=increase_volume).grid(sticky="E", row=5)

master.mainloop()
