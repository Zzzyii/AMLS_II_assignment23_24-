
# Exploratory Data Analysis
# Show most used top n data in datasets
def get_n(df, field, n, top=True):
    top_graphemes = df.groupby([field]).size().reset_index(name='counts')['counts'].sort_values(ascending=not top)[:n]
    top_grapheme_roots = top_graphemes.index
    top_grapheme_counts = top_graphemes.values
    top_graphemes = class_map[class_map['component_type'] == field].reset_index().iloc[top_grapheme_roots]
    top_graphemes.drop(['component_type', 'label'], axis=1, inplace=True)
    top_graphemes.loc[:, 'count'] = top_grapheme_counts
    return top_graphemes

# Calculate the number of different data values
def plot_count(feature, title, df, size=1):

    with sns.axes_style("whitegrid"):
        f, ax = plt.subplots(1,1, figsize=(4*size,4))
        total = float(len(df))
        g = sns.countplot(df[feature], order = df[feature].value_counts().index[:10], palette='Set3')
        g.set_title("Percentage of {}".format(title))
        if(size > 2):
            plt.xticks(rotation=90, size=8)
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{:1.2f}%'.format(100*height/total),
                    ha="center")
        plt.show()

#Rescales a DataFrame containing an image to a specified size.
def resize(df, size=64, need_progress_bar=True):
    resized = {}
    resize_size=64
    if need_progress_bar:
        for i in tqdm(range(df.shape[0])):
            image=df.loc[df.index[i]].values.reshape(137,236)
            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

            idx = 0
            ls_xmin = []
            ls_ymin = []
            ls_xmax = []
            ls_ymax = []
            for cnt in contours:
                idx += 1
                x,y,w,h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x + w)
                ls_ymax.append(y + h)
            xmin = min(ls_xmin)
            ymin = min(ls_ymin)
            xmax = max(ls_xmax)
            ymax = max(ls_ymax)

            roi = image[ymin:ymax,xmin:xmax]
            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)
            resized[df.index[i]] = resized_roi.reshape(-1)
    else:
        for i in range(df.shape[0]):
            image=df.loc[df.index[i]].values.reshape(137,236)
            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]

            idx = 0
            ls_xmin = []
            ls_ymin = []
            ls_xmax = []
            ls_ymax = []
            for cnt in contours:
                idx += 1
                x,y,w,h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x + w)
                ls_ymax.append(y + h)
            xmin = min(ls_xmin)
            ymin = min(ls_ymin)
            xmax = max(ls_xmax)
            ymax = max(ls_ymax)

            roi = image[ymin:ymax,xmin:xmax]
            resized_roi = cv2.resize(roi, (resize_size, resize_size),interpolation=cv2.INTER_AREA)
            resized[df.index[i]] = resized_roi.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized

#Convert all category variables in a dataframe into dummy variables.
def get_dummies(df):
    cols = []
    for col in df:
        cols.append(pd.get_dummies(df[col].astype(str)))
    return pd.concat(cols, axis=1)

# Create model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten
from tensorflow.keras.models import Model

# Convolution, batch normalisation, ReLU activation
def conv_bn_relu(inputs, filters, kernel_size, strides=(1, 1)):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

#Basic residual blocks
def basic_block(inputs, filters, strides=(1, 1), downsample=False):
    residual = inputs
    x = conv_bn_relu(inputs, filters, kernel_size=(3, 3), strides=strides)
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)

    if downsample:
        residual = Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, padding="same")(inputs)
        residual = BatchNormalization()(residual)

    x = Add()([x, residual])
    x = Activation("relu")(x)
    return x

# Created a ResNet34 model with multiple output
def build_resnet34(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=1000):
    inputs = Input(shape=input_shape)

    x = conv_bn_relu(inputs, filters=64, kernel_size=(7, 7), strides=(2, 2))
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

    filters = 64
    for i, blocks in enumerate([3, 4, 6, 3]):  
        for block in range(blocks):
            if block == 0 and i != 0:
                x = basic_block(x, filters=filters, strides=(2, 2), downsample=True)
            else:
                x = basic_block(x, filters=filters)
        filters *= 2

    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)

    # Multiple output layer
    head_root = Dense(168, activation='softmax', name='root')(x)
    head_vowel = Dense(11, activation='softmax', name='vowel')(x)
    head_consonant = Dense(7, activation='softmax', name='consonant')(x)

    model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])

    return model

# A custom data generator is defined and produces multiple labelled batches of data simultaneously, suitable for training multi-output models.
class DataGenerator(keras.preprocessing.image.ImageDataGenerator):

    def flow(self,
             x,
             y=None,
             batch_size=32,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             subset=None):

        targets = None
        target_lengths = {}
        ordered_outputs = []
        for output, target in y.items():
            if targets is None:
                targets = target
            else:
                targets = np.concatenate((targets, target), axis=1)
            target_lengths[output] = target.shape[1]
            ordered_outputs.append(output)


        for flowx, flowy in super().flow(x, targets, batch_size=batch_size,
                                         shuffle=shuffle):
            target_dict = {}
            i = 0
            for output in ordered_outputs:
                target_length = target_lengths[output]
                target_dict[output] = flowy[:, i: i + target_length]
                i += target_length

            yield flowx, target_dict


#Iterative plotting of accuracy and loss for training and validation sets.
def train_loss(his, epoch, title):
    plt.style.use('classic')
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history['root_loss'], label='train_root_loss')
    plt.plot(np.arange(0, epoch), his.history['vowel_loss'], label='train_vowel_loss')
    plt.plot(np.arange(0, epoch), his.history['consonant_loss'], label='train_consonant_loss')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Train_Loss')
    plt.legend(loc='upper right')
    plt.show()

def val_loss(his, epoch, title):
    plt.style.use('classic')
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history['val_root_loss'], label='val_root_loss')
    plt.plot(np.arange(0, epoch), his.history['val_vowel_loss'], label='val_vowel_loss')
    plt.plot(np.arange(0, epoch), his.history['val_consonant_loss'], label='val_consonant_loss')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Val_Loss')
    plt.legend(loc='upper right')
    plt.show()

def train_acc(his, epoch, title):
    plt.style.use('classic')
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history['root_accuracy'], label='train_root_acc')
    plt.plot(np.arange(0, epoch), his.history['vowel_accuracy'], label='train_vowel_accuracy')
    plt.plot(np.arange(0, epoch), his.history['consonant_accuracy'], label='train_consonant_accuracy')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Train_Accuracy')
    plt.legend(loc='lower right')
    plt.show()

def val_acc(his, epoch, title):
    plt.style.use('classic')
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history['val_root_accuracy'], label='val_root_acc')
    plt.plot(np.arange(0, epoch), his.history['val_vowel_accuracy'], label='val_vowel_accuracy')
    plt.plot(np.arange(0, epoch), his.history['val_consonant_accuracy'], label='val_consonant_accuracy')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Val_Accuracy')
    plt.legend(loc='lower right')
    plt.show()