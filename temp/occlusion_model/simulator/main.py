import os
import cv2
import config
import shutil

from plant_generator import generate_plant

dir = 'test'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.mkdir(dir)

for n in range(50):
    img = generate_plant(config.name)
    img_name = f'simulated_image_{n + 1}.png'
    cv2.imwrite(os.path.join(dir, img_name), img)

# img = generate_plant(config.name)
# img_name = f'simulated_image.png'
# cv2.imwrite(img_name, img)

# ana = rb.SegmentAnalyser()
# for i, rs in enumerate(allRS):
#       vtpname = "../results/example_2b_" + str(i) + ".vtp"
#       rs.write(vtpname)
#       c_a = rb.SegmentAnalyser(rs)
#       ana.addSegments(c_a)
#
# ana.write("../results/example_2b_all.vtp")
