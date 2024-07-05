最近在鲲鹏920上搭建了tensorflow环境，运行Model Zoo中的image_classification里的models时需要准备ILSVRC2012数据集，运行maskrcnn里的models时需要准备coco2017。所以本仓库记录从零准备数据集的过程。

注：由于目的是为了让模型跑通，所以可能数据集的处理有些不合适，但最终结果是模型顺利eval。

[toc]

# 下载models

处理ILSVRC2012 coco2017需要用到tensorflow-models(master)中的文件，所以需要 `git clone https://github.com/tensorflow/models.git`。

# ILSVRC2012

参考链接：https://blog.csdn.net/RayChiu757374816/article/details/126870264

## 下载并解压数据集

[官网](https://image-net.org/index.php) or [百度网盘](https://pan.baidu.com/s/1AxBChFN1rckQLfCOv-nsKw?pwd=3bzf  提取码：3bzf)：ILSVRC2012_bbox_val_v3.tgz和ILSVRC2012_img_val.tar

```
tar -xvf ILSVRC2012_img_val.tar -C val/
tar -xvf ILSVRC2012_bbox_val_v3.tgz -C bbox/
```

## 修改build_imagenet_data.py

### step1-修改数据路径(100行左右)

```
# tf.app.flags.DEFINE_string('train_directory', '/tmp/',
#                            'Training data directory')
# tf.app.flags.DEFINE_string('validation_directory', '/tmp/',
#                            'Validation data directory')
# tf.app.flags.DEFINE_string('output_directory', '/tmp/',
#                            'Output data directory')

tf.app.flags.DEFINE_string('train_directory', '/home/tf-test/dataset/ILSVRC2012',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/home/tf-test/dataset/ILSVRC2012',
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', '/home/tf-test/dataset/ILSVRC2012',
                           'Output data directory')
```

### step2-修改range的返回类型(500行左右)

```
#shuffled_index = range(len(filenames))
shuffled_index = list(range(len(filenames)))
```

### step3-修改bytes(217行左右)

```
# colorspace = 'RGB'
# channels = 3
# image_format = 'JPEG'

colorspace = b'RGB'
channels = 3
image_format = b'JPEG'
```

### step4-修改读写方式(317行左右)

```
#image_data = tf.gfile.GFile(filename, 'r').read()
image_data = tf.gfile.GFile(filename, 'rb').read()
```

### step5-匹配python3(175行左右)

```
# def _bytes_feature(value):
#   """Wrapper for inserting bytes features into Example proto."""
#   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  if(type(value) is not bytes):
    value=value.encode("utf8")
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
```

### step6-修改np.int(451行左右)

```
#spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(int)
```

### step7-适配placeholder(246行左右)

```
# def __init__(self):
#   # Create a single Session to run all image coding calls.
#   self._sess = tf.Session()
  
#   # Initializes function that converts PNG to JPEG data.
#   self._png_data = tf.placeholder(dtype=tf.string)
#   image = tf.image.decode_png(self._png_data, channels=3)
#   self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

#   # Initializes function that converts CMYK JPEG data to RGB JPEG data.
#   self._cmyk_data = tf.placeholder(dtype=tf.string)
#   image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
#   self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

#   # Initializes function that decodes RGB JPEG data.
#   self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
#   self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

def __init__(self):
  # Create a single Session to run all image coding calls.
  self._sess = tf.Session()

  tf.compat.v1.disable_eager_execution()
  # Initializes function that converts PNG to JPEG data.
  self._png_data = tf.placeholder(dtype=tf.string)
  image = tf.image.decode_png(self._png_data, channels=3)
  self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

  # Initializes function that converts CMYK JPEG data to RGB JPEG data.
  self._cmyk_data = tf.placeholder(dtype=tf.string)
  image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
  self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

  # Initializes function that decodes RGB JPEG data.
  self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
  self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
```

## 数据处理

工作路径：/home/tf-test/dataset/models/research/slim/datasets

```
python preprocess_imagenet_validation_data.py /home/tf-test/dataset/ILSVRC2012/val/ imagenet_2012_validation_synset_labels.txt
python process_bounding_boxes.py /home/tf-test/dataset/ILSVRC2012/bbox/ imagenet_lsvrc_2015_synsets.txt | sort > imagenet_2012_bounding_boxes.csv
python build_imagenet_data.py --output_directory /home/tf-test/dataset/ILSVRC2012/imagenet_tf/ --validation_directory /home/tf-test/dataset/ILSVRC2012/val/
```

## 查看结果

```
[tf-test@jiakai-openeuler-01 imagenet_tf]$ ls
train-00000-of-01024  train-00231-of-01024  train-00462-of-01024  train-00693-of-01024  train-00924-of-01024
train-00001-of-01024  train-00232-of-01024  train-00463-of-01024  train-00694-of-01024  train-00925-of-01024
train-00002-of-01024  train-00233-of-01024  train-00464-of-01024  train-00695-of-01024  train-00926-of-01024
...
```

# coco2017

## 下载并解压数据集

[annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) [val2017.zip](http://images.cocodataset.org/zips/val2017.zip) [instances_val2017.json](https://github.com/eembc/mlmark/blob/main/datasets/COCO2017/annotations/instances_val2017.json)

```
unzip val2017.zip
unzip annotations_trainval2017.zip
```

## 数据处理

工作路径：/home/tf-test/dataset/models/official/vision/data

```
 python create_coco_tf_record.py --logtostderr \
      --image_dir=/home/tf-test/dataset/coco2017/val2017/ \
      --caption_annotations_file=/home/tf-test/dataset/coco2017/annotations/captions_val2017.json \
      --output_file_prefix=/home/tf-test/dataset/coco2017/output/val \
      --num_shards=100
```

## 查看结果

```
[tf-test@jiakai-openeuler-01 output]$ ls
val-00000-of-00100.tfrecord  val-00025-of-00100.tfrecord  val-00050-of-00100.tfrecord  val-00075-of-00100.tfrecord
val-00001-of-00100.tfrecord  val-00026-of-00100.tfrecord  val-00051-of-00100.tfrecord  val-00076-of-00100.tfrecord
val-00002-of-00100.tfrecord  val-00027-of-00100.tfrecord  val-00052-of-00100.tfrecord  val-00077-of-00100.tfrecord
val-00003-of-00100.tfrecord  val-00028-of-00100.tfrecord  val-00053-of-00100.tfrecord  val-00078-of-00100.tfrecord
...
```

注：instances_val2017.json要放在output下，与val*文件放在一起
