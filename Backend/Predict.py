#!/usr/bin/env python
# coding: utf-8

# In[4]:


from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import tensorflow as tf



# In[5]:


model = tf.keras.models.load_model('/Users/gunjeetbawa/Smart Scan/Backend/model_vgg14.h5')


# In[7]:


#img = image.load_img('val/NORMAL/NORMAL2-IM-1430-0001.jpeg', target_size=(224, 224))
img = image.load_img('/Users/gunjeetbawa/Smart Scan/Backend/val/PNEUMONIA/person1946_bacteria_4874.jpeg', target_size=(224, 224))
checker = image.img_to_array(img)
checker = np.expand_dims(checker, axis=0)
img_data = preprocess_input(checker)
classes = model.predict(img_data)
#print(img_data)

print(classes)

classes = [1,0]
if classes[0]==1:
    print("Normal")
else:
    print("Pse")


# In[ ]:





# In[ ]:




