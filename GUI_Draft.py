from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, Frame, filedialog, Label, font

# defining a path for icons and assets
OUTPUT_PATH = Path("./assets").parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


# Main application class
class MainApplication(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.mainwindow = GUI(self)
        self.mainwindow.frames()
        self.mainwindow.buttons()
        self.mainwindow.pack(side="right", fill="both", expand=True)


# Main frontend class
class GUI(Frame):
    def frames(self):
        self.toolbar_frame = Frame(bg = "#BEDFDD",
                        width = main_app.winfo_screenwidth(),
                        bd = 1,
                        highlightthickness = 0,
                        relief = "ridge")
        self.toolbar_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.toolbar_frame.grid_rowconfigure(1, weight=0)
        self.toolbar_frame.grid_columnconfigure(7, weight=1)
        self.toolbar_frame.grid_propagate(1)
        app_label = Label(self.toolbar_frame, bg = "#BEDFDD", text = "Sleezy", font=font.Font(family = "gothic", size = 10))
        app_label.grid(row=1, column=1, padx=10, pady=10)

        self.frame_up = Frame(main_app, bg = "#D4E5E4",
                        height = main_app.winfo_screenheight()/1.5,
                        width = main_app.winfo_screenwidth(),
                        bd = 3,
                        highlightthickness = 0,
                        relief = "ridge")
        self.frame_up.grid(row=1, column=0, columnspan=2, sticky="nsew")

        self.frame_down_left = Frame(main_app, bg = "#BEDFDD",
                        height = main_app.winfo_screenheight()/4,
                        width = main_app.winfo_screenwidth()/2,
                        bd = 3,
                        highlightthickness = 0,
                        relief = "ridge")
        self.frame_down_left.grid(row=2, column = 0, columnspan=1, sticky="ew")

        self.frame_down_right = Frame(main_app, bg = "#BEDFDD",
                        height = main_app.winfo_screenheight()/4,
                        width = main_app.winfo_screenheight()/2,
                        bd = 3,
                        highlightthickness = 0,
                        relief = "ridge")
        self.frame_down_right.grid(row=2, column = 1, columnspan=1, sticky="ew")

    def buttons(self):
        self.button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
        button_1 = Button(self.toolbar_frame, image=self.button_image_1,
                        borderwidth=1,
                        highlightthickness=0,
                        command=lambda: backend.button1_click(),
                        relief="flat")
        button_1.grid(row=1, column=2, sticky="ew", padx=5, pady=10)

        self.button_image_2 = PhotoImage(file=relative_to_assets("button_2.png"))
        button_2 = Button(self.toolbar_frame, image=self.button_image_2,
                        borderwidth=1,
                        highlightthickness=0,
                        command=lambda: backend.button2_click(),
                        relief="flat")
        button_2.grid(row=1, column=3, sticky="ew", padx=5, pady=10)

        self.button_image_3 = PhotoImage(file=relative_to_assets("button_3.png"))
        button_3 = Button(self.toolbar_frame, image=self.button_image_3,
                        borderwidth=1,
                        highlightthickness=0,
                        command=lambda: backend.button3_click(),
                        relief="flat")
        button_3.grid(row=1, column=4, sticky="ew", padx=5, pady=10)

        self.button_image_4 = PhotoImage(file=relative_to_assets("button_4.png"))
        button_4 = Button(self.toolbar_frame, image=self.button_image_4,
                        borderwidth=1,
                        highlightthickness=0,
                        command=lambda: backend.button4_click(),
                        relief="flat")
        button_4.grid(row=1, column=5, sticky="ew", padx=5, pady=10)

        self.button_image_5 = PhotoImage(file=relative_to_assets("button_5.png"))
        button_5 = Button(self.toolbar_frame, image=self.button_image_5,
                        borderwidth=2,
                        highlightthickness=0,
                        command=lambda: backend.button5_click(),
                        relief="flat")
        button_5.grid(row=1, column=6, sticky="ew", padx=5, pady=10)

        empty_frame = Frame(self.toolbar_frame, bg = "#BEDFDD",
                        bd = 0,
                        highlightthickness = 0,
                        relief = "ridge")
        empty_frame.grid(row=1, column=7)

        self.button_image_6 = PhotoImage(file=relative_to_assets("button_6.png"))
        button_6 = Button(self.toolbar_frame, image=self.button_image_6,
                        borderwidth=1,
                        highlightthickness=0,
                        command=lambda: backend.button6_click(),
                        relief="flat")
        button_6.grid(row=1, column=8, sticky="ew", padx=5, pady=10)

        self.button_image_7 = PhotoImage(file=relative_to_assets("button_7.png"))
        button_7 = Button(self.toolbar_frame, image=self.button_image_7,
                        borderwidth=1,
                        highlightthickness=0,
                        command=lambda: backend.button7_click(),
                        relief="flat")
        button_7.grid(row=1, column=9, sticky="ew", padx=5, pady=10)

        self.button_image_8 = PhotoImage(file=relative_to_assets("button_8.png"))
        button_8 = Button(self.toolbar_frame, image=self.button_image_8,
                        borderwidth=1,
                        highlightthickness=0,
                        command=lambda: backend.button8_click(),
                        relief="flat")
        button_8.grid(row=1, column=10, sticky="ew", padx=5, pady=10)

        self.button_image_9 = PhotoImage(file=relative_to_assets("button_9.png"))
        button_9 = Button(self.toolbar_frame, image=self.button_image_9,
                        borderwidth=1,
                        highlightthickness=0,
                        command=lambda: backend.button9_click(),
                        relief="flat")
        button_9.grid(row=1, column=11, sticky="ew", padx=5, pady=10)
    
# Main backend class
class backend:
    def button1_click():
        return filedialog.askopenfile(mode='r', filetypes = '')

    def button2_click():
        print ("Button 2 Clicked")

    def button3_click():
        print ("Button 3 Clicked")

    def button4_click():
        print ("Button 4 Clicked")

    def button5_click():
        print ("Button 5 Clicked")

    def button6_click():
        print ("Button 6 Clicked")

    def button7_click():
        print ("Button 7 Clicked")

    def button8_click():
        print ("Button 8 Clicked")

    def button9_click():
        print ("Button 9 Clicked")
        
if __name__ == "__main__":
    main_app = Tk()
    main_app.title("Sleezy")
    main_app.resizable(True, True)   
    main_app.geometry("%dx%d" % (main_app.winfo_screenwidth(), main_app.winfo_screenheight()))
    main_app.configure(bg = "#D4F6E4")
    main_app.grid_rowconfigure(1, weight=1)
    main_app.grid_columnconfigure(1, weight=1)
    load_app = MainApplication(master=main_app)
    load_app.mainloop()
