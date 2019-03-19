import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

hand = mpimg.imread("data/5.jpg")

print("hand.jpg shape: ", hand.shape)

'''
plt.imshow(qeleng_hand)
plt.show()
plt.clf()
'''

hand_hsv = cv2.cvtColor(hand, cv2.COLOR_RGB2HSV)

h = hand_hsv[:, :, 0]
#print(np.max(h))
s = hand_hsv[:, :, 1]
#print(np.max(s))
v = hand_hsv[:, :, 2]
#print(np.max(v))



f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,5))
ax1.set_title("H")
ax1.imshow(h)
ax2.set_title("S")
ax2.imshow(s)
ax3.set_title("V")
ax3.imshow(v)
plt.show()

plt.clf()

#print(qh_hsv[500,1000,:])


low = []
high = []

def restrict(color_component):
    if color_component < 0:
        return 0
    elif color_component > 255:
        return 255
    else:
        return color_component

z_value = 5.5

for i in range(3):
    lines_0_to_1000 = np.array(hand_hsv[0:1000, :, i]).flatten()
    lines_3200_to_4000 = np.array(hand_hsv[3200:4000, :, i]).flatten()
    top_and_bottom = pd.DataFrame(np.append(lines_0_to_1000, lines_3200_to_4000))
    mu = top_and_bottom.values.mean()
    sigma = top_and_bottom.values.std()
    deviation = z_value*sigma
    low.append(restrict(mu-deviation))
    high.append(restrict(mu+deviation))
    #print(top_and_bottom.head())
    #print("\n{}\n".format(top_and_bottom.describe()))

'''
line_3500_v = pd.DataFrame(qh_hsv[3500, :, 2])
print("line_3500_v", line_3500_v.describe())
'''

print("\n\nlow:", low)

print("\n\nhigh:", high)

mask_lower = np.array([low[0], low[1], low[2]])
mask_higher = np.array([high[0], high[1], high[2]])

hand_mask = cv2.inRange(hand_hsv, mask_lower, mask_higher)
plt.imshow(hand_mask, cmap="gray")
plt.show()

masked_hand = np.copy(hand)
masked_hand[hand_mask != 0] = [0, 0, 0]

plt.imshow(masked_hand)
plt.show()

masked_hand[:1000,:,:] = [0, 0, 0]
masked_hand[3200:,:,:] = [0, 0, 0]

plt.imshow(masked_hand)
plt.show()

background = mpimg.imread("qalang_wallpaper.jpg")
print(background.shape)
plt.imshow(background)
plt.show()

bg_cropped = background[:len(hand), :len(hand[0]), :]
plt.imshow(bg_cropped)
plt.show()

print(hand_mask.shape)
hand_mask[:1000,:] = [1]
hand_mask[3200:,:] = [1]

bg_masked = np.copy(bg_cropped)
bg_masked[hand_mask == 0] = [0,0,0]

plt.imshow(bg_masked)
plt.show()

full_picture = bg_masked + masked_hand

plt.imshow(full_picture)
plt.show()

