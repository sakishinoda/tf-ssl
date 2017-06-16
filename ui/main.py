import tkinter as tk
from PIL import Image, ImageTk

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.grid(padx=10, pady=10)
        self.createWidgets()
        self.showImage()

    def createWidgets(self):

        self.canvas = tk.Canvas(self, width=256, height=256, bd=5)
        self.canvas.grid(row=0, column=1, columnspan=2)
        self.canvas.bind("<Button-1>", self.addLabel)

        self.quitButton = tk.Button(self, text='Quit', command=self.quit)
        self.quitButton.grid(row=1, column=2)
        self.saveButton = tk.Button(self, text='Save')
        self.saveButton.grid(row=1, column=1)

        # Set up label log table
        self.labelTable = tk.Text(self, height=30, width=30)
        self.labelTable.grid(row=2, columnspan=2)
        self.labelTable.insert('end', 'X\tY\tTag\n')

    def showImage(self):
        photo = ImageTk.PhotoImage(Image.open('sample.jpeg'))
        self.canvas.create_image(0, 0, anchor='nw', image=photo)
        self.canvas.current_image = photo

    def addLabel(self, event):
        # Specify size of marker
        rad = 5

        x1, y1 = (event.x - rad), (event.y - rad)
        x2, y2 = (event.x + rad), (event.y + rad)

        # Create horizontal line
        self.canvas.create_line(x1, event.y, x2, event.y, fill='blue')
        # Create vertical line
        self.canvas.create_line(event.x, y1, event.x, y2, fill='blue')

        ## Create dot
        # self.canvas.create_oval(x1, y1, x2, y2, fill='blue')

        # Find coordinates in image and print to table
        canvas = event.widget
        x = canvas.canvasx(event.x)
        y = canvas.canvasy(event.y)
        tag = canvas.find_closest(x, y)
        self.labelTable.insert('end', '{}\t{}\t{}\n'.format(x,y,tag))

    def save(self):
        # Get coordinates of all labels and write to file as metadata for image tile
        return None




app = Application()
app.master.title('Termite Mound Labeller')
app.mainloop()


