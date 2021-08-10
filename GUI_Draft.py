from pathlib import Path
import tkinter as tk
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage

# defining a path for icons and assets
OUTPUT_PATH = Path('/media/ubuntu/casper-rw/upper/home/ubuntu/build/assets').parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

# Main window class
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.widgets()
    
    def run(self):
        self.mainloop()
    
    #widgets of the main window
    def widgets(self):
        #main background
        canvas = Canvas(window,
                        bg = "#FFFFFF",
                        height = 850,
                        width = 1440,
                        bd = 0,
                        highlightthickness = 0,
                        relief = "ridge")
        # background image control
        canvas.place(x = 0, y = 0)
        canvas.create_rectangle(0.0,
                                0.0,
                                1440.0,
                                850.0,
                                fill="#D4E5E4",
                                outline="")
        # toolbar creation
        canvas.create_rectangle(0.0,
                                0.0,
                                1440.0,
                                44.0,
                                fill="#BEDFDD",
                                outline="")
        # Software name and logo when ready
        # Sleezy as a primary naem :)
        canvas.create_text(11.0,
                            13.0,
                            anchor="nw",
                            text="Sleezy",
                            fill="#136262",
                            font=("NovaFlat", 14 * -1))
        
        canvas.create_rectangle(64.0,
                                7.0,
                                64.0,
                                37.0,
                                fill="#000000",
                                outline="")

        canvas.create_rectangle(200.0,
                                7.0,
                                200.0,
                                37.0,
                                fill="#000000",
                                outline="")
        
        # creating the buttons as they are in the wirefarme
        button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
        button_1 = Button(image=button_image_1,
                            borderwidth=0,
                            highlightthickness=0,
                            command=lambda: print("button_1 clicked"),
                            relief="flat")
        button_1.place(x=82.0,
                        y=9.0,
                        width=25.0,
                        height=25.0)

        button_image_2 = PhotoImage(file=relative_to_assets("button_2.png"))
        button_2 = Button(image=button_image_2,
                        borderwidth=0,
                        highlightthickness=0,
                        command=lambda: print("button_2 clicked"),
                        relief="flat")
        button_2.place(x=120.0,
                        y=9.0,
                        width=25.0,
                        height=25.0)

        button_image_3 = PhotoImage(file=relative_to_assets("button_3.png"))
        button_3 = Button(image=button_image_3,
            borderwidth=0,
            highlightthickness=0,
            command=lambda: print("button_3 clicked"),
            relief="flat")
        button_3.place(x=158.0,
                        y=9.0,
                        width=25.0,
                        height=25.0)

        button_image_4 = PhotoImage(file=relative_to_assets("button_4.png"))
        button_4 = Button(image=button_image_4,
                    borderwidth=0,
                    highlightthickness=0,
                    command=lambda: print("button_4 clicked"),
                    relief="flat")
        button_4.place(x=218.0,
                        y=9.0,
                        width=25.0,
                        height=25.0)

        button_image_5 = PhotoImage(file=relative_to_assets("button_5.png"))
        button_5 = Button(image=button_image_5,
                        borderwidth=0,
                        highlightthickness=0,
                        command=lambda: print("button_5 clicked"),
                        relief="flat")
        button_5.place(x=256.0,
                        y=9.0,
                        width=25.0,
                        height=25.0)
        
        button_image_6 = PhotoImage(file=relative_to_assets("button_6.png"))
        button_6 = Button(image=button_image_6,
                        borderwidth=0,
                        highlightthickness=0,
                        command=lambda: print("button_6 clicked"),
                        relief="flat")
        button_6.place(x=1269.0,
                        y=9.0,
                        width=25.0,
                        height=25.0)

        button_image_7 = PhotoImage(file=relative_to_assets("button_7.png"))
        button_7 = Button(image=button_image_7,
                        borderwidth=0,
                        highlightthickness=0,
                        command=lambda: print("button_7 clicked"),
                        relief="flat")
        button_7.place(x=1326.0,
                        y=9.0,
                        width=25.0,
                        height=25.0)

        button_image_8 = PhotoImage(file=relative_to_assets("button_8.png"))
        button_8 = Button(image=button_image_8,
                        borderwidth=0,
                        highlightthickness=0,
                        command=lambda: print("button_8 clicked"),
                        relief="flat")
        button_8.place(x=1364.0,
                        y=9.0,
                        width=25.0,
                        height=25.0)

        button_image_9 = PhotoImage(file=relative_to_assets("button_9.png"))
        button_9 = Button(image=button_image_9,
                        borderwidth=0,
                        highlightthickness=0,
                        command=lambda: print("button_9 clicked"),
                        relief="flat")
        button_9.place(x=1402.0,
                        y=9.0,
                        width=25.0,
                        height=25.0)

        # outlining other spaces for other tools in the wireframe
        canvas.create_rectangle(9.0,
                                50.0,
                                1431.0,
                                219.0,
                                fill="#FFFFFF",
                                outline="")

        canvas.create_rectangle(9.0,
                                234.0,
                                1431.0,
                                403.0,
                                fill="#FFFFFF",
                                outline="")

        canvas.create_rectangle(9.0,
                                418.0,
                                1431.0,
                                587.0,
                                fill="#FFFFFF",
                                outline="")

        canvas.create_rectangle(13.0,
                                605.0,
                                653.0,
                                836.0,
                                fill="#BEDFDD",
                                outline="")

        canvas.create_rectangle(666.0,
                                605.0,
                                1431.0,
                                836.0,
                                fill="#BEDFDD",
                                outline="")

        canvas.create_rectangle(1310.0,
                                7.0,
                                1310.0,
                                37.0,
                                fill="#000000",
                                outline="")
    
    # defining button functions as we progress and replacing them with the above lambda funstions
    def button1_click():
        print ('Button 1 Clicked')
        
if __name__ == '__main__':
    window = Tk()
    window.geometry("1440x850")
    window.configure(bg = "#FFFFFF")
    app = Application(master=window)
    app.run()
