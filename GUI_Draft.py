from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, Frame, filedialog, Label

# defining a path for icons and assets
OUTPUT_PATH = Path("/media/ubuntu/casper-rw/upper/home/ubuntu/build/assets").parent
ASSETS_PATH = OUTPUT_PATH / Path("./assets")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


# Main application class
class MainApplication(Frame):
    def __init__(self, master):
        super().__init__(master)
        #self.reportbar = Reportbar(self)
        self.toolbar = Toolbar(self)
        self.toolbar.widgets.frames(self)
        self.toolbar.widgets.buttons(self)
        #self.navigationbar = Navigationbar(self)
        self.mainwindow = MainWindow(self)
        self.mainwindow.widgets.frames(self)

        #self.reportbar.pack(side="bottom", fill="x")
        self.toolbar.pack(side="top", fill="x", expand=True)
        #self.navigationbar.pack(side="left", fill="y")
        self.mainwindow.pack(side="right", fill="both", expand=True)

class Toolbar(Frame):
    class widgets:
        def frames(self):
            toolbar_height = 40
            toolbar_frame = Frame(bg = "#BEDFDD",
                            height = toolbar_height,
                            width = main_app.winfo_screenwidth(),
                            bd = 1,
                            highlightthickness = 0,
                            relief = "ridge")
            toolbar_frame.place(x = 0, y = 0)
            
            # Software name and logo when ready
            # Sleezy as a primary na:)
            '''canvas.create_text(11.0,
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
                                    outline="")'''
            
        def buttons(self):
            # creating the buttons as they are in the wireframe
            self.button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
            button_1 = Button(image=self.button_image_1,
                            borderwidth=1,
                            highlightthickness=0,
                            command=MainWindow.functions.button1_click,
                            relief="flat")
            button_1.place(x=82.0,
                            y=9.0,
                            width=25.0,
                            height=25.0)

            self.button_image_2 = PhotoImage(file=relative_to_assets("button_2.png"))
            button_2 = Button(image=self.button_image_2,
                            borderwidth=1,
                            highlightthickness=0,
                            command=MainWindow.functions.button2_click,
                            relief="flat")
            button_2.place(x=120.0,
                            y=9.0,
                            width=25.0,
                            height=25.0)

            self.button_image_3 = PhotoImage(file=relative_to_assets("button_3.png"))
            button_3 = Button(image=self.button_image_3,
                            borderwidth=1,
                            highlightthickness=0,
                            command=MainWindow.functions.button3_click,
                            relief="flat")
            button_3.place(x=158.0,
                            y=9.0,
                            width=25.0,
                            height=25.0)

            self.button_image_4 = PhotoImage(file=relative_to_assets("button_4.png"))
            button_4 = Button(image=self.button_image_4,
                            borderwidth=1,
                            highlightthickness=0,
                            command=MainWindow.functions.button4_click,
                            relief="flat")
            button_4.place(x=218.0,
                            y=9.0,
                            width=25.0,
                            height=25.0)

            self.button_image_5 = PhotoImage(file=relative_to_assets("button_5.png"))
            button_5 = Button(image=self.button_image_5,
                            borderwidth=2,
                            highlightthickness=0,
                            command=MainWindow.functions.button5_click,
                            relief="flat")
            button_5.place(x=256.0,
                            y=9.0,
                            width=25.0,
                            height=25.0)

            self.button_image_6 = PhotoImage(file=relative_to_assets("button_6.png"))
            button_6 = Button(image=self.button_image_6,
                            borderwidth=1,
                            highlightthickness=0,
                            command=MainWindow.functions.button6_click,
                            relief="flat")
            button_6.place(x=main_app.winfo_screenwidth()-180,
                            y=9.0,
                            width=25.0,
                            height=25.0)

            self.button_image_7 = PhotoImage(file=relative_to_assets("button_7.png"))
            button_7 = Button(image=self.button_image_7,
                            borderwidth=1,
                            highlightthickness=0,
                            command=MainWindow.functions.button7_click,
                            relief="groove")
            button_7.place(x=main_app.winfo_screenwidth()-120,
                            y=9.0,
                            width=25.0,
                            height=25.0)

            self.button_image_8 = PhotoImage(file=relative_to_assets("button_8.png"))
            button_8 = Button(image=self.button_image_8,
                            borderwidth=1,
                            highlightthickness=0,
                            command=MainWindow.functions.button8_click,
                            relief="groove")
            button_8.place(x=main_app.winfo_screenwidth()-80,
                            y=9.0,
                            width=25.0,
                            height=25.0)

            self.button_image_9 = PhotoImage(file=relative_to_assets("button_9.png"))
            button_9 = Button(image=self.button_image_9,
                            borderwidth=1,
                            highlightthickness=0,
                            command=MainWindow.functions.button9_click,
                            relief="groove")
            button_9.place(x=main_app.winfo_screenwidth()-40,
                            y=9.0,
                            width=25.0,
                            height=25.0)        
        
        
        
# Main window class
class MainWindow(Frame):
    
    #widgets of the main window
    class widgets:
        
        def frames(self):
            frame_up = Frame(bg = "#D4E5E4",
                            height = main_app.winfo_screenheight()/1.66,
                            width = main_app.winfo_screenwidth(),
                            bd = 3,
                            highlightthickness = 0,
                            relief = "ridge")
            frame_up.place(x = 0, y = 40)
            
            frame_down_left = Frame(bg = "#BEDFDD",
                            height = main_app.winfo_screenheight(),
                            width = main_app.winfo_screenwidth()/2,
                            bd = 3,
                            highlightthickness = 0,
                            relief = "ridge")
            frame_down_left.place(x = 0, y = main_app.winfo_screenheight()/1.66 + 40)
            
            frame_down_right = Frame(bg = "#BEDFDD",
                            height = 650,
                            width = main_app.winfo_screenheight(),
                            bd = 3,
                            highlightthickness = 0,
                            relief = "ridge")
            frame_down_right.place(x = main_app.winfo_screenwidth()/2, y = main_app.winfo_screenheight()/1.66 + 40)
          
            
        def canvases(self):
            #main background

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
    class functions:
        def button1_click(self):
            return filedialog.askopenfile(mode='r', filetypes = '')
        
        def button2_click(self):
            print ("Button 2 Clicked")

        def button3_click(self):
            print ("Button 3 Clicked")

        def button4_click(self):
            print ("Button 4 Clicked")

        def button5_click(self):
            print ("Button 5 Clicked")

        def button6_click(self):
            print ("Button 6 Clicked")

        def button7_click(self):
            print ("Button 7 Clicked")

        def button8_click(self):
            print ("Button 8 Clicked")

        def button9_click(self):
            print ("Button 9 Clicked")
        
if __name__ == "__main__":
    main_app = Tk()
    main_app.title("Sleezy")
    main_app.resizable(True, True)   
    main_app.geometry("%dx%d" % (main_app.winfo_screenwidth(), main_app.winfo_screenheight()))
    main_app.configure(bg = "#D4F6E4")
    load_app = MainApplication(master=main_app)
    load_app.mainloop()
