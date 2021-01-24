from flask import Flask, request,render_template
from flask_cors import CORS,cross_origin
import matplotlib
import shutil
import numpy as np
import os
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
matplotlib.use('Agg')
sys.path.append("..")
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__) # initializing a flask app
CORS(app)

MODEL_NAME = 'my_model_satellite_detection_mask_rcnn'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'labelmapsat.pbtxt')

# loading tf graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# loading classes file
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)



@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    
    return render_template("index.html")

# @app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
# @cross_origin()
# def predict():
#     print('in new predict')
#     return render_template("results.html")


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def predict():
    #image = request.json['image']
    # print(image)
    # imgdata = base64.b64decode(image)
    uploaded_file = request.files['upload_file']
    destination = './static/inputImage.jpg'
    filename = uploaded_file.filename
    uploaded_file.save(filename)
    target = './' + str(filename)
    shutil.copy(target, destination)
    # with open(filename, 'wb') as f:
    #     f.write(imgdata)
    #     f.close()
    print('after img right')
    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    IMAGE_SIZE = (12, 8)

    def run_inference_for_single_image(image, graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    image_path='./static/inputImage.jpg'
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.savefig('./static/output.jpg')
    print('near return')
    #return results()
    # try:
    #     print('in try')
    #     return render_template('results.html')
    # except Exception as e:
    #     return "Something Went Wrong....Unable Please try again."
    return render_template('results.html')
    print('after return')
    #return send_file('./static/output.jpg')
    #return redirect('./templates/results.html')

# @app.route('/results',methods=['POST','GET']) # route to show the images on a webpage
# @cross_origin()
# def results():
#     print('in results')
#     if request.method == 'POST':
#         print('in post')
#         return render_template("results.html")
#
#     else:
#         return "something went wrong"
# #     # return render_template('show_image.html')



if __name__ == "__main__":
    #to run on cloud
    #port = int(os.getenv("PORT"))
    #app.run(host='0.0.0.0', port=port)  # running the app

    #to run locally
    app.run(host='127.0.0.1', port=8000, debug=True)
    #app.run(host='0.0.0.0', port=8000, debug=True)



