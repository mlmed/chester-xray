/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

function display_size_data_full(){
  // Check for support of the PerformanceResourceTiming.*size properties and print their values
  // if supported.
  if (performance === undefined) {
    console.log("= Display Size Data: performance NOT supported");
    return;
  }

  var list = performance.getEntriesByType("resource");
  if (list === undefined) {
    console.log("= Display Size Data: performance.getEntriesByType() is  NOT supported");
    return;
  }

  // For each "resource", display its *Size property values
  console.log("= Display Size Data");
  for (var i=0; i < list.length; i++) {
	if (list[i].name.includes("shard")){
    console.log("== Resource[" + i + "] - " + list[i].name);
    if ("decodedBodySize" in list[i])
      console.log("... decodedBodySize[" + i + "] = " + list[i].decodedBodySize);
    else
      console.log("... decodedBodySize[" + i + "] = NOT supported");

    if ("encodedBodySize" in list[i])
      console.log("... encodedBodySize[" + i + "] = " + list[i].encodedBodySize);
    else
      console.log("... encodedBodySize[" + i + "] = NOT supported");

    if ("transferSize" in list[i])
      console.log("... transferSize[" + i + "] = " + list[i].transferSize);
    else
      console.log("... transferSize[" + i + "] = NOT supported");
    }
  }
}

function display_size_data(){
  // Check for support of the PerformanceResourceTiming.*size properties and print their values
  // if supported.
  if (performance === undefined) {
    console.log("= Display Size Data: performance NOT supported");
    return;
  }

  var list = performance.getEntriesByType("resource");
  if (list === undefined) {
    console.log("= Display Size Data: performance.getEntriesByType() is  NOT supported");
    return;
  }

  // For each "resource", display its *Size property values
  var todo = 0
  var total = 0
  for (var i=0; i < list.length; i++) {
	if (list[i].name.includes("shard")){
    	total+=1
    }
  }
  //console.log('Loading model... ' + total + "/" + 59)
  status('Loading model... ' + total + "/" + 8);
}

const MODEL_PATH = 'chest2';

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 10;
let mobilenet;

const mobilenetDemo = async () => {
  status('Loading model...');
  var t=setInterval(display_size_data,100);
  mobilenet = await tf.loadFrozenModel(MODEL_PATH + "/tensorflowjs_model.pb", MODEL_PATH + "/weights_manifest.json");
  clearInterval(t);
  //mobilenet = await tf.loadModel(MOBILENET_MODEL_PATH); // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.

  //mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
  mobilenet.predict(tf.zeros([1, 3, IMAGE_SIZE, IMAGE_SIZE])).dispose();
  status(''); // Make a prediction through the locally hosted cat.jpg.

  const catElement = document.getElementById('cat');

  if (catElement.complete && catElement.naturalHeight !== 0) {
    predict(catElement);
    catElement.style.display = '';
  } else {
    catElement.onload = () => {
      predict(catElement);
      catElement.style.display = '';
    };
  }

  document.getElementById('file-container').style.display = '';
};
/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */


async function predict(imgElement) {
  status('Predicting...');
  const startTime = performance.now();
  output = tf.tidy(() => {
    // tf.fromPixels() returns a Tensor from an image element.
	  
    const img = tf.fromPixels(imgElement).toFloat();
    const offset = tf.scalar(255); // Normalize the image from [0, 255] to [-1, 1].

    const normalized = img.div(offset);

    const batched = normalized.mean(2).reshape([1, 1, IMAGE_SIZE, IMAGE_SIZE]).tile([1, 3,1,1])
    //const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]); // Make a prediction through mobilenet.
    //const batched = normalized.mean(2).reshape([1, 3, IMAGE_SIZE, IMAGE_SIZE]); // Make a prediction through mobilenet.
    
    const result = mobilenet.execute(batched, ["Relu", "Relu_1", "Sigmoid"])

    return result
    
    //b = a.reshape([64, 112, 112]).mean(0)

    //return mobilenet.predict(batched);
  });

  layer1 = output[0].mean(0).mean(0) //112, 112
  
  layer2 = output[1].mean(0).mean(0) // 56. 56
  
  logits = await output[2].data()
  
  console.log(logits)

  const classes = await distOverClasses(logits)
  
  var c = document.getElementById("layer1");
  tf.toPixels(layer1.div(layer1.max()),c)
  
  var c = document.getElementById("layer2");
  tf.toPixels(layer2.div(layer2.max()),c)

  
  
// plot layer first as text
//  for (x = 0; x < 56; x++) {
//	s = ""
//	for (y = 0; y < 56; y++) {
//		if (layer2[x*56+y]>0.1){
//		s+="x  "
//        }else{
//		s+="   "
//        }
//    }
//    console.log(s)
//  }
  
  const totalTime = performance.now() - startTime;
  status(`Done in ${Math.floor(totalTime)}ms`); // Show the classes in the DOM.

  showResults(imgElement, classes);
}
/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */

async function distOverClasses(values){
	
	//values = await logits.data();
	
	pathologies = ["Atelectasis", "Consolidation", "Infiltration",
        "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
        "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"]
	
	values = values.subarray(0,pathologies.length)
	
//	softmax = await tf.softmax(values).data()
//	out = await tf.argMax(softmax).data()
	
	const topClassesAndProbs = [];
	for (let i = 0; i < values.length; i++) {
	    topClassesAndProbs.push({
	      className: pathologies[i],
	      probability: values[i]
	    });
	}
	return topClassesAndProbs
}



async function getTopKClasses(logits, topK) {
  const values = await logits.data();
  const valuesAndIndices = [];

  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({
      value: values[i],
      index: i
    });
  }

  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);

  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];

  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: IMAGENET_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    });
  }

  return topClassesAndProbs;
}

function decimalToHexString(number){
  if (number < 0){
    number = 0xFFFFFFFF + number + 1;
  }
  return number.toString(16).toUpperCase();
}

function showResults(imgElement, classes) {
  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';
  const imgContainer = document.createElement('div');
  imgContainer.appendChild(imgElement);
  predictionContainer.appendChild(imgContainer);
  const probsContainer = document.createElement('div');

  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('div');
    row.className = 'row';
    const classElement = document.createElement('div');
    classElement.className = 'cell';
    classElement.innerText = classes[i].className;
    row.appendChild(classElement);
    const probsElement = document.createElement('div');
    probsElement.className = 'cell';
    probsElement.innerText = classes[i].probability.toFixed(3);
    scale = parseInt((1-classes[i].probability)*255)
    probsElement.style.backgroundColor = "rgb(255," + scale + "," + scale + ")";
    row.appendChild(probsElement);
    probsContainer.appendChild(row);
  }

  predictionContainer.appendChild(probsContainer);
  predictionsElement.insertBefore(predictionContainer, predictionsElement.firstChild);
}

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files; // Display thumbnails & issue call to predict each image.

  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }

    let reader = new FileReader();
    const idx = i; // Closure to capture the file information.

    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;

      img.onload = () => predict(img);
    }; // Read in the image file as a data URL.


    reader.readAsDataURL(f);
  }
});
const demoStatusElement = document.getElementById('status');

const status = msg => demoStatusElement.innerText = msg;

const predictionsElement = document.getElementById('predictions');
mobilenetDemo();
