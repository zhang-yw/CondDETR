# import os
 
# html = ""

# path = "/Users/yw-zhang/Downloads/5_cross_att_layer"
# fname = "index.html"

# for file in os.listdir(path):
# 	html += f"<center><img src='{file}'/ width=100% ></center><br>"
 
# with open(os.path.join(path, fname), "w") as outputfile:
# 	outputfile.write(html)
 

import os
 
html = ""

path = "/Users/yw-zhang/Downloads/vis/1"
fname = "index.html"

filenames = ['000000427338.jpg', '000000424975.jpg', '000000120853.jpg', '000000099810.jpg', '000000370818.jpg', '000000016502.jpg', '000000416256.jpg', '000000338905.jpg', '000000423617.jpg',  '000000295138.jpg', '000000066523.jpg', '000000031269.jpg', '000000014439.jpg', '000000453584.jpg', '000000009914.jpg', '000000210230.jpg', '000000306136.jpg', '000000263425.jpg', '000000288042.jpg', '000000396526.jpg', '000000016598.jpg', '000000323799.jpg', '000000159282.jpg', '000000474078.jpg', '000000564280.jpg', '000000175387.jpg', '000000223959.jpg', '000000492110.jpg', '000000186345.jpg', '000000106757.jpg', '000000495732.jpg', '000000495054.jpg', '000000218249.jpg', '000000537964.jpg', '000000050165.jpg', '000000163746.jpg', '000000020247.jpg', '000000500565.jpg', '000000287527.jpg', '000000365207.jpg', '000000068833.jpg', '000000499181.jpg', '000000521141.jpg', '000000434996.jpg', '000000281179.jpg', '000000214200.jpg']

for filename in filenames:
	html = ""
	html += f"<center><img src='../1_cross_att_layer/{filename}'/ width=100% ></center><br>"
	html += f"<center><img src='../1_cross_att_final/{filename}'/ width=100% ></center><br>"
	html += f"<center><img src='../1_cross_att_head/{filename}'/ width=100% ></center><br>"
	with open(os.path.join(path, filename), "w") as outputfile:
		outputfile.write(html)
 