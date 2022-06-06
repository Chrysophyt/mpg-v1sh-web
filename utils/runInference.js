import { runSqueezenetModel } from './modelHelper';
import * as Jimp from 'jimp';
import { Tensor } from 'onnxruntime-web';

function imageDataToTensor(JimpImageFile){
    // 1. Get buffer data from image and create R, G, and B arrays.
    var imageBufferData = JimpImageFile.bitmap.data;
    const redArray = new Array();
    const greenArray = new Array();
    const blueArray = new Array();
  
    // 2. Loop through the image buffer and extract the R, G, and B channels
    for (let i = 0; i < imageBufferData.length; i += 4) {
      redArray.push(imageBufferData[i]);
      greenArray.push(imageBufferData[i + 1]);
      blueArray.push(imageBufferData[i + 2]);
      // skip alpha channel
    }
  
    // 3. Concatenate RGB to transpose [224, 224, 3] -> [3, 224, 224] to a number array
    const transposedData = redArray.concat(greenArray).concat(blueArray);
  
    // 4. convert to float32
    let i, l = transposedData.length; // length, we need this for the loop
    // create the Float32Array size 3 * 224 * 224 for these dimensions output
    const float32Data = new Float32Array(dims[1] * dims[2] * dims[3]);
    for (i = 0; i < l; i++) {
      float32Data[i] = transposedData[i] / 255.0; // convert to float
    }
    // 5. create the tensor object from onnxruntime-web.
    const inputTensor = new Tensor("float32", float32Data, dims);
    return inputTensor;
  }
  

export async function inferenceSqueezenet(imageFile){
    return new Promise((resolve, reject) => {
        // 1. Convert image to tensor
        const imageTensor = await toImageTensor(imageFile);
        // 2. Run model
        const [predictions, inferenceTime] = await runSqueezenetModel(imageTensor);
        // 3. Return predictions and the amount of time it took to inference.
        resolve([predictions, inferenceTime])
      })
    }