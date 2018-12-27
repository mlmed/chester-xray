function display_size_data_full(){
  if (performance === undefined) {
    console.log("= Display Size Data: performance NOT supported");
    return;
  }

  var list = performance.getEntriesByType("resource");
  if (list === undefined) {
    console.log("= Display Size Data: performance.getEntriesByType() is  NOT supported");
    return;
  }

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
  if (performance === undefined) {
    console.log("= Display Size Data: performance NOT supported");
    return;
  }

  var list = performance.getEntriesByType("resource");
  if (list === undefined) {
    console.log("= Display Size Data: performance.getEntriesByType() is  NOT supported");
    return;
  }
  
  var todo = 0
  var total = 0
  for (var i=0; i < list.length; i++) {
	if (list[i].name.includes("shard")){
    	total+=1
    }
  }
  status('Loading model... ' + total + "/" + 8);
}

const MODEL_PATH = 'chestxnet1';

const IMAGE_SIZE = 224;
const TOPK_PREDICTIONS = 10;
let mobilenet;

const run = async () => {
	status('Loading model...');
	var t=setInterval(display_size_data,100);
	mobilenet = await tf.loadFrozenModel(MODEL_PATH + "/tensorflowjs_model.pb", MODEL_PATH + "/weights_manifest.json");
	status('Loading model into memory');
	clearInterval(t);

	mobilenet.predict(tf.zeros([1, 3, IMAGE_SIZE, IMAGE_SIZE])).dispose();
	status('');

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

async function predict(imgElement) {
	status('Predicting...');
	const startTime = performance.now();
	
	output = tf.tidy(() => {
	  
	    const img = tf.fromPixels(imgElement).toFloat();
	
	    const normalized = img.div(tf.scalar(255));
	
	    const batched = normalized.mean(2).reshape([1, 1, IMAGE_SIZE, IMAGE_SIZE]).tile([1, 3,1,1])
	    
	    const result = mobilenet.execute(batched, ["Sigmoid", "Relu", "Relu_1", "Relu_3", "Relu_14"])
	
	    return result
    
	});
  
	logits = await output[0].data()

	layers = []
  
	layers.push(["layer1",output[1].mean(0).abs().mean(0)])
	layers.push(["layer2",output[2].mean(0).abs().mean(0)])
	layers.push(["layer3",output[3].mean(0).abs().mean(0)])
	layers.push(["layer4",output[4].mean(0).abs().mean(0)])
  
	//console.log(logits)

	const classes = await distOverClasses(logits)

	const totalTime = performance.now() - startTime;
	status(`Done in ${Math.floor(totalTime)}ms`); // Show the classes in the DOM.

	showResults(imgElement, layers, classes);
}

async function distOverClasses(values){
	
	pathologies = ["Atelectasis", "Consolidation", "Infiltration",
        "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
        "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"]
	
	values = values.subarray(0,pathologies.length)
	
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

function showResults(imgElement, layers, classes) {
	
	const predictionContainer = document.createElement('div');
	predictionContainer.className = 'pred-container';
	const imgContainer = document.createElement('div');
	imgContainer.appendChild(imgElement);
	predictionContainer.appendChild(imgContainer);
	
	const layersContainer = document.createElement('div');
  
	for(i = 0; i < layers.length; i++){
		layerName = layers[i][0]
		layer = layers[i][1]
		var c = document.createElement('canvas');
		tf.toPixels(layer.div(layer.max()),c)
		c.style.width = "112px"
		c.style.imageRendering = "pixelated"
		layersContainer.appendChild(c);
	}
		
	predictionContainer.appendChild(layersContainer);
  
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
    const idx = i;

    reader.onload = e => {
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;

      img.onload = () => predict(img);
    }; 


    reader.readAsDataURL(f);
  }
});
const statusElement = document.getElementById('status');

const status = msg => statusElement.innerText = msg;

const predictionsElement = document.getElementById('predictions');


function findGetParameter(parameterName) {
    var result = null,
        tmp = [];
    location.search
        .substr(1)
        .split("&")
        .forEach(function (item) {
          tmp = item.split("=");
          if (tmp[0] === parameterName) result = decodeURIComponent(tmp[1]);
        });
    return result;
}

$("#agree").click(function(){
	$("#agree").hide()
	run();
});


