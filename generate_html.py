import os
 
html = ""

path = "/Users/yw-zhang/Downloads/5_cross_att_layer"
fname = "index.html"

for file in os.listdir(path):
	html += f"<center><img src='{file}'/ width=100% ></center><br>"
 
with open(os.path.join(path, fname), "w") as outputfile:
	outputfile.write(html)
 