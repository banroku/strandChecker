train_datagen = ImageDataGenerator(
        rescale=1./2,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=30,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./2)

train_generator = train_datagen.flow_from_directory(
        'image/image_train',
        target_size=(224, 224),
        batch_size=4,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'image/image_cv',
        target_size=(224, 224),
        batch_size=4,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=100)
