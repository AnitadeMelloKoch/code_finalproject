from curses import window
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from src.model.stageII.stageII_gan import StageIIGAN
from src.utils.training_utils import adjust_image

gan = StageIIGAN('runs/final/')
gan.load()

root= tk.Tk()

canvas1 = tk.Canvas(root, width = 1000, height = 800,  relief = 'raised')
canvas1.pack()

label1 = tk.Label(root, text='StackGAN trained on flower images.')
label1.config(font=('helvetica', 22))
window1 = canvas1.create_window(200, 25, window=label1)
canvas1.move(window1, 300, 0)

label2 = tk.Label(root, text='Type your Caption:')
label2.config(font=('helvetica', 16))
window2 = canvas1.create_window(200, 100, window=label2)
canvas1.move(window2, 300, 0)

entry1 = tk.Entry (root, width=50, font=('helvetica', 16)) 
window3 = canvas1.create_window(200, 140, window=entry1)
canvas1.move(window3, 300, 0)


low_img = ImageTk.PhotoImage(image=Image.fromarray(np.zeros((64,64,3), dtype=np.uint8)))
high_img = ImageTk.PhotoImage(image=Image.fromarray(np.zeros((256,256,3), dtype=np.uint8)))

label3 = tk.Label(root, text="", font=('helvetica', 16))
window4 = canvas1.create_window(200, 210, window=label3)
canvas1.move(window4, 300, 0)
image_obj_low = canvas1.create_image(400, 400, image=low_img)
image_obj_high = canvas1.create_image(600, 400, image=high_img)

def get_flower ():
    
    x1 = entry1.get()

    global high_img
    global low_img

    low_res, high_res = gan.evaluate(x1)
    low_res = adjust_image(low_res)
    high_res = adjust_image(high_res)
    high_img = ImageTk.PhotoImage(image=Image.fromarray((high_res*255).astype(np.uint8)))
    low_img = ImageTk.PhotoImage(image=Image.fromarray((low_res*255).astype(np.uint8)))
    
    label3.config(text=x1)
    canvas1.itemconfig(image_obj_high, image=high_img)
    canvas1.itemconfig(image_obj_low, image=low_img)
    
button1 = tk.Button(text='Generate Image', command=get_flower, bg='brown', fg='white', font=('helvetica', 9, 'bold'))
button_window = canvas1.create_window(200, 180, window=button1)
canvas1.move(button_window, 300,0)

root.mainloop()
