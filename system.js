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

prog = "\\"
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
  if (prog == "\\"){
	  prog="-";
  }else if (prog == "-"){
	  prog="/";
  }else{
	  prog="\\";
  }
  status('Loading model... ' + total + "/" + (8+16) + "  " + prog);
}

function sleep(ms) {
	  return new Promise(resolve => setTimeout(resolve, ms));
}

const IMAGE_SIZE = 224;
const RECSCORE_THRESH = 0.5;
const OODSCORE_THRESH = 1000;

let mobilenet;
let chestgrad;
let mobileaenet;
let catElement;
let grad_fns;
let img;
//chestxnet1
//const LABELS = ["Atelectasis", "Consolidation", "Infiltration",
//    "Pneumothorax", "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
//   "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"]
//chestxnet2
const LABELS = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'];

const OP_POINT = [0.45879191, 0.20330566, 0.34361544, 0.30163303, 0.50299263,
    0.36888129, 0.29530331, 0.6088959 , 0.46361208, 0.17098247,
    0.31575406, 0.51793754, 0.49182123, 0.59332716];

async function run(){
	
	try{
		await run_real()
	}catch(err) {
		clearInterval(downloadStatus);
		$(".loading").hide()
		status("Error! " + err.message);
		console.log(err)
		
		if (err.message == "Failed to fetch"){
			status("Error! Failed to fetch the neural network models. Try disabling your ad blocker as this may prevent models from being downloaded from a different domain. (There are no ads here)");
		}

	}
}

let downloadStatus
async function run_real(){
	status('Loading model...');
	const startTime = performance.now();
	downloadStatus=setInterval(display_size_data,100);
	const MODEL_PATH = 'https://mlmed.github.io/tools/xray/models/chestxnet2';
	mobilenet = await tf.loadFrozenModel(MODEL_PATH + "/tensorflowjs_model.pb", MODEL_PATH + "/weights_manifest.json");
	console.log("First Model loaded " + Math.floor(performance.now() - startTime) + "ms");
	const AEMODEL_PATH = 'https://mlmed.github.io/tools/xray/models/chestae1';
	mobileaenet = await tf.loadFrozenModel(AEMODEL_PATH + "/tensorflowjs_model.pb", AEMODEL_PATH + "/weights_manifest.json");
	console.log("Second Model loaded " + Math.floor(performance.now() - startTime) + "ms");
	clearInterval(downloadStatus);
//	status('Loading gradients...');
//	
//	grad_fns = []
//	for (var i = 0; i < 3; i++) {
//		console.log(i)
//		grad_fn = tf.grad(x => mobilenet.predict(x).squeeze().slice(i,1));
//		grad_fns.push(grad_fn)
//	}
	
	status('Loading model into memory...');
	
	await sleep(100)
	
	chestgrad = tf.grad(x => mobilenet.predict(x))
	
//	mobilenet.predict(tf.zeros([1, 3, IMAGE_SIZE, IMAGE_SIZE])).dispose();
//	mobileaenet.predict(tf.zeros([1, 1, 64, 64])).dispose();
	status('');

	catElement = document.getElementById('cat');

	if (catElement.complete && catElement.naturalHeight !== 0) {
		predict(catElement, "Example Image (" + catElement.src.substring(catElement.src.lastIndexOf('/')+1)+ ")");
	} else {
		catElement.onload = () => {
			predict(catElement, "Example Image (" + catElement.src.substring(catElement.src.lastIndexOf('/')+1)+ ")");
		};
	}

	document.getElementById('file-container').style.display = '';
};

let batched
let grads
let currentpred
async function predict(imgElement, name) {
	
	try{
		$("#file-container #files").attr("disabled", true)
		$(".computegrads").each((k,v) => {v.style.display = "none"});
		
		const startTime = performance.now();
		await predict_real(imgElement, name);
		
		$(".loading").each((k,v) => {v.style.display = "none"});
		const totalTime = performance.now() - startTime;
		status(`Done in ${Math.floor(totalTime)}ms`);
		
	}catch(err) {
		$(".loading").hide()
		status("Error! " + err.message);
		console.log(err)
	}
	$("#file-container #files").attr("disabled", false)
}
async function predict_real(imgElement, name) {
	status('Predicting...');
	
	const startTime = performance.now();
	w = imgElement.width
	h = imgElement.height
	if (w < h){
		imgElement.width = IMAGE_SIZE
		imgElement.height = Math.floor(IMAGE_SIZE*h/w)
	}else{
		imgElement.height = IMAGE_SIZE
		imgElement.width = Math.floor(IMAGE_SIZE*w/h)
	}
	
	
	console.log("img wxh: " + w + ", " + h + " => " + imgElement.width + ", " + imgElement.height)
	
	currentpred = $("#predtemplate").clone();
	currentpred.find(".loading").each((k,v) => {v.style.display = "block"});
	currentpred[0].id = ""
	predictionsElement.insertBefore(currentpred[0], predictionsElement.firstChild);
	
	//currentpred.find(".inputimage").attr("src", imgElement.src)
	currentpred[0].style.display="block";
	
	currentpred.find(".imagename").text(name)
	
    img = tf.fromPixels(imgElement).toFloat();
	
    normalized = img.div(tf.scalar(255));

    meanImg = normalized.mean(2)
    hOffset = Math.floor(img.shape[1]/2 - IMAGE_SIZE/2)
    wOffset = Math.floor(img.shape[0]/2 - IMAGE_SIZE/2)
    
    cropImg = meanImg.slice([wOffset,hOffset],[IMAGE_SIZE,IMAGE_SIZE])
    
	//////// display input image
	imgs = currentpred.find(".inputimage")
	for (i=0; i < imgs.length; i++){
		canvas = imgs[i]

		await tf.toPixels(cropImg,canvas);	
		canvas.style.width = "100%";
		canvas.style.height = "";
		canvas.style.imageRendering = "pixelated";
	}
	////////////////////
    
    batched = cropImg.reshape([1, 1, IMAGE_SIZE, IMAGE_SIZE]).tile([1,3,1,1])
	
    console.log("Prepared input image " + Math.floor(performance.now() - startTime) + "ms");
    
	
	status('Computing Reconstruction...');
	
	img_small = document.createElement('img');
	img_small.src = imgElement.src
	img_small.width = 64
	img_small.height = 64
	
	let {recInput, recErr, rec} = tf.tidy(() => {
		
	    const img = tf.fromPixels(img_small).toFloat();
		
	    const normalized = img.div(tf.scalar(255));
	
	    const batched = normalized.mean(2).reshape([1, 1, 64, 64])
	    
	    const batched2 = batched.mul(2).sub(1)
	    
	    const rec = mobileaenet.predict(batched)
	
	    const recErr = batched.sub(rec).abs()
	    
	    return {recInput:batched, recErr: recErr, rec: rec};
	});
	
	recScore = recErr.mean().dataSync()
	console.log(recScore);
	
	console.log("Computed Reconstruction " + Math.floor(performance.now() - startTime) + "ms");
	
	
	canvas_a = currentpred.find(".inputimage_rec")[0]
	layer = recInput.reshape([64,64])
	await tf.toPixels(layer.div(2).add(0.5),canvas_a);
	
	canvas_b = currentpred.find(".recimage")[0]
	layer = rec.reshape([64,64])
	await tf.toPixels(layer.div(2).add(0.5),canvas_b);
	
	console.log("Wrote images " + Math.floor(performance.now() - startTime) + "ms");
	
	// compute ssim
	canvas = canvas_a
	a = {width: canvas.width, height: canvas.height, data: canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data, channels: 4, canvas: canvas}
	canvas = canvas_b
	b = {width: canvas.width, height: canvas.height, data: canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data, channels: 4, canvas: canvas}
	
	// https://github.com/darosh/image-ssim-js
	ssim = ImageSSIM.compare(a, b, 8, 0.01, 0.03, 8)
	console.log("ssim " + JSON.stringify(ssim));
	
	console.log("Computed SSIM " + Math.floor(performance.now() - startTime) + "ms");
	
	//////// display ood image

	canvas = currentpred.find(".oodimage")[0]
	layer = recErr.reshape([64,64])
	await tf.toPixels(layer.div(2),canvas);
	canvas.style.width = "100%";
	canvas.style.height = "";
	canvas.style.imageRendering = "pixelated";
	
	ctx = canvas.getContext("2d");
	d = ctx.getImageData(0, 0, canvas.width, canvas.height);
	makeColor(d.data);
	makeTransparent(d.data)
	ctx.putImageData(d,0,0);
	
	scoreBox = document.createElement("center")
	score = "recScore:" + parseFloat(recScore).toFixed(2)  + ", ssim:" + ssim.ssim.toFixed(2) 
	scoreBox.innerText = score
	currentpred.find(".oodimagebox")[0].append(scoreBox)
	
	currentpred.find(".oodviz .loading")[0].style.display = "none";
	currentpred.find(".oodimagebox")[0].style.display = "block";
	////////////////////
	
	console.log("Plotted Reconstruction " + Math.floor(performance.now() - startTime) + "ms");


	// zoom does not work yet
//	var main = currentpred.find(".inputimage")[0]
//	var zoom = document.getElementById("zoom");
//	var ctx = main.getContext("2d")
//	var zoomCtx = zoom.getContext("2d");
//
//	main.addEventListener("mousemove", function(e){
//	    console.log(e);
//	    zoomCtx.fillStyle = "white";
//	    zoomCtx.fillRect(0,0, zoom.width, zoom.height);
//	    zoomCtx.drawImage(main, (e.x-$(main).offset().left)/2, e.y-$(main).offset().top, 100, 50, 0,0, 400, 200);
//	    zoom.style.top = e.pageY + 10 + "px"
//	    zoom.style.left = e.pageX + 10 + "px"
//	    zoom.style.display = "block";
//	});
//
//	main.addEventListener("mouseout", function(){
//	    zoom.style.display = "none";
//	});
	
	
//  m = img.mean(2);
//  oodscore = 
//  img.slice([0,0,0],[-1,-1,1]).sub(m.expandDims(2)).abs().sum().add(
//  img.slice([0,0,1],[-1,-1,1]).sub(m.expandDims(2)).abs().sum()).add( 
//  img.slice([0,0,2],[-1,-1,1]).sub(m.expandDims(2)).abs().sum()).dataSync();
//  console.log("oodscore " + oodscore);
//	else if (oodscore > OODSCORE_THRESH){
//		
//		showProbErrorColor(currentpred.find(".predbox")[0], recScore)
//		return

	
	
	
 
	status('Predicting disease...');
	await sleep(100)
    
	can_predict = ssim.ssim > 0.30
	
	if (!can_predict){
		
		
		showProbError(currentpred.find(".predbox")[0], score)
		return
		
		
	}else{
		output = tf.tidy(() => {
		 
			return mobilenet.execute(batched, ["Sigmoid"])
		});
	    
		logits = await output.data()
	
		console.log("Computed logits and grad " + Math.floor(performance.now() - startTime) + "ms");
		console.log("logits=" + logits)
		
		const classes = await distOverClasses(logits)
		
		showProbResults(currentpred.find(".predbox")[0], classes, recScore)
		currentpred.find(".predviz .loading")[0].style.display = "none";
		
	
		currentpred.find(".gradviz .loading").hide()
		if (can_predict){
			currentpred.find(".gradviz .computegrads").show()
			
			currentpred.find(".gradviz .computegrads").click(function(){
				currentpred.find(".gradviz .computegrads").hide()
				computeGrads(currentpred, batched);
			});
		}
		
	
		console.log("results plotted " + Math.floor(performance.now() - startTime) + "ms");
	}
	
}

async function computeGrads(thispred, batched){

	try{
		status('Computing gradients...');
		$("#file-container #files").attr("disabled", true)
		
		const startTime = performance.now();
		await computeGrads_real(thispred, batched);
		
		const totalTime = performance.now() - startTime;
		status(`Done in ${Math.floor(totalTime)}ms`);
		
	}catch(err) {
		$(".loading").hide()
		status("Error! " + err.message);
		console.log(err)
	}
	
	$("#file-container #files").attr("disabled", false)
}

async function computeGrads_real(thispred, batched){
	
	thispred.find(".gradviz .loading")[0].style.display = "block";
	
	await sleep(100)
	
	grad = tf.tidy(() => {
		
	    const temp = mobilenet.executor.checkTensorForDisposal
	    mobilenet.executor.checkTensorForDisposal=function(){}
	    
		const grad = chestgrad(batched);
	    
	    mobilenet.executor.checkTensorForDisposal = temp
	    
	    return grad
    
	});
	
	//////// display grad image
	canvas = thispred.find(".gradimage")[0]
	layer = grad.mean(0).abs().max(0)
	await tf.toPixels(layer.div(layer.max()),canvas);	
	canvas.style.width = "100%";
	canvas.style.height = "";
	canvas.style.imageRendering = "pixelated";
	
	ctx = canvas.getContext("2d");
	d = ctx.getImageData(0, 0, canvas.width, canvas.height);
	makeColor(d.data);
	makeTransparent(d.data)
	ctx.putImageData(d,0,0);
	
	thispred.find(".gradviz .loading")[0].style.display = "none";
	thispred.find(".gradimagebox")[0].style.display = "block";
	////////////////////
	
}


async function distOverClasses(values){
	
	pathologies = LABELS
	
	values = values.subarray(0,pathologies.length)
	
	const topClassesAndProbs = [];
	for (let i = 0; i < values.length; i++) {
		
		if (values[i]<OP_POINT[i]){
			value_normalized = values[i]/(OP_POINT[i]*2)
		}else{
			value_normalized = 1-((1-values[i])/((1-(OP_POINT[i]))*2))
			
		}
		console.log(pathologies[i] + ",pred:" + values[i] + "," + "OP_POINT:" + OP_POINT[i] + "->normalized:" + value_normalized)

	    topClassesAndProbs.push({
	      className: pathologies[i],
	      probability: value_normalized
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


function invertColors(data) {
  for (var i = 0; i < data.length; i+= 4) {
    data[i] = data[i] ^ 255; // Invert Red
    data[i+1] = data[i+1] ^ 255; // Invert Green
    data[i+2] = data[i+2] ^ 255; // Invert Blue
  }
}

function enforceBounds(x) {
    if (x < 0) {
        return 0;
    } else if (x > 1){
        return 1;
    } else {
        return x;
    }
}
function interpolateLinearly(x, values) {
    // Split values into four lists
    var x_values = [];
    var r_values = [];
    var g_values = [];
    var b_values = [];
    for (i in values) {
        x_values.push(values[i][0]);
        r_values.push(values[i][1][0]);
        g_values.push(values[i][1][1]);
        b_values.push(values[i][1][2]);
    }
    var i = 1;
    while (x_values[i] < x) {
        i = i+1;
    }
    i = i-1;
    var width = Math.abs(x_values[i] - x_values[i+1]);
    var scaling_factor = (x - x_values[i]) / width;
    // Get the new color values though interpolation
    var r = r_values[i] + scaling_factor * (r_values[i+1] - r_values[i])
    var g = g_values[i] + scaling_factor * (g_values[i+1] - g_values[i])
    var b = b_values[i] + scaling_factor * (b_values[i+1] - b_values[i])
    return [enforceBounds(r), enforceBounds(g), enforceBounds(b)];
}


function makeColor(data) {
	
	  for (var i = 0; i < data.length; i+= 4) {
		var color = interpolateLinearly(data[i]/255, jet);
	    data[i] = Math.round(255*color[0]); // Invert Red
	    data[i+1] = Math.round(255*color[1]); // Invert Green
	    data[i+2] = Math.round(255*color[2]); // Invert Blue
	  }
	}
function makeTransparent(pix) {
	//var imgd = ctx.getImageData(0, 0, imageWidth, imageHeight),
	//pix = imgd.data;
	
	for (var i = 0, n = pix.length; i <n; i += 4) {
		var r = pix[i],
		    g = pix[i+1],
		    b = pix[i+2];
		
		if(g < 20){ 
		    // If the green component value is higher than 150
		    // make the pixel transparent because i+3 is the alpha component
		    // values 0-255 work, 255 is solid
		    pix[i + 3] = 0;
		}
	}
	//ctx.putImageData(imgd, 0, 0);​
}



function showProbError(predictionContainer, score) {
	
	const row = document.createElement('div');
	row.className = 'row';
	row.style.width="100%"
	row.textContent = "This image is too far out of our training distribution so we will not process it. (" + score + "). It could be that your image is not cropped correctly or it was aquired using a protocal that is not in our training data. "
	predictionContainer.appendChild(row);
}

function showProbErrorColor(predictionContainer) {
	
	const row = document.createElement('div');
	row.className = 'row';
	row.style.width="100%"
	row.textContent = "This image appears to be a color image and we suspect it is not an xray."
	predictionContainer.appendChild(row);
}

function showProbResults(predictionContainer, classes) {
		
	const probsContainer = document.createElement('div');
	probsContainer.style.width="100%";
	probsContainer.style.minWidth="220px";
	
	for (let i = -1; i < classes.length; i++) {
	    const row = document.createElement('div');
	    row.className = 'row';
	    const classElement = document.createElement('div');
	    classElement.className = 'cell';
	    
	    const probsElement = document.createElement('div');
	    if (i == -1){
	    	classElement.innerText = "Name";
	    	classElement.style.fontSize="x-small";
	    	probsElement.className = 'cell';
	    }else{
	    	classElement.innerText = classes[i].className;
	    	if (classes[i].probability > 0.6){
	    		classElement.style.fontWeight = "900";
	    	}
	    	classElement.style.fontSize="small";
	    	probsElement.className = 'cell gradient';
	    	probsElement.style.borderBottomStyle="solid";
	    	probsElement.style.borderColor = "white";
	    	probsElement.style.borderWidth="2";
	    }
	    row.appendChild(classElement);
	    
	    
	    
	    probsElement.style.width="100%";
	    probsElement.style.textAlign="left";
	    probsElement.style.position="relative";
	    
	    if (i == -1){
		    target = document.createElement('span');
		    target.innerText = "Healthy";
		    target.style.position="absolute";
		    target.style.fontSize="x-small"
		    target.style.left=0;
		    probsElement.appendChild(target)
		    
		    target = document.createElement('span');
		    target.innerText = "Risk";
		    target.style.position="absolute";
		    target.style.fontSize="x-small"
		    target.style.right=0;
		    probsElement.appendChild(target)
		    
		    
	    }else{
		    target = document.createElement('span');
		    //target.innerText = "|";
		    target.className="glyphicon glyphicon-asterisk"
		    target.style.marginLeft="-7px"; //glyh is 14x14
		    target.style.position="absolute";
		    target.style.left=parseInt(classes[i].probability*100) + "%";
		    target.style.fontWeight="900";
		    probsElement.appendChild(target)
	    }
	    
	    //probsElement.innerText = (parseInt(classes[i].probability*100)) + "%";
	    //scale = parseInt((1-classes[i].probability)*255)
	    //probsElement.style.backgroundColor = "rgb(255," + scale + "," + scale + ")";
	    row.appendChild(probsElement);
	    probsContainer.appendChild(row);
	    
	}

	predictionContainer.appendChild(probsContainer);
	
	$(".gradient").hover(
		function(e){
			a=$(this).find("span")[0];
			a.innerHTML=a.style.left
//			if (parseInt(a.style.left)>60){
//				a.style.marginLeft="-30px";
//			}
		},
		function(e){
			a=$(this).find("span")[0];
			a.innerHTML="";
		},
	);
}


async function showResults(imgElement, layers, classes, recScore) {
	
	const predictionContainer = document.createElement('div');
	predictionContainer.className = 'row';
	const imgContainer = document.createElement('div');
	imgContainer.className="col-xs-3";
	imgElement.style.width = "100%";
	imgElement.height = "auto";
	imgElement.style.height = "auto";
	imgElement.style.display = "";
	imgContainer.appendChild(imgElement);
	predictionContainer.appendChild(imgContainer);
	
	const layersContainer = document.createElement('div');
	layersContainer.className="col-xs-6";
	for(i = 0; i < layers.length; i++){
		layerName = layers[i][0];
		layer = layers[i][1];
		var canvas = document.createElement('canvas');
		await tf.toPixels(layer.div(layer.max()),canvas);		
		canvas.style.width = "100%";
		canvas.style.height = "";
		canvas.style.imageRendering = "pixelated";
		const layerBox = document.createElement('span');
		layerBox.appendChild(canvas);
		
		ctx = canvas.getContext("2d");
		d = ctx.getImageData(0, 0, canvas.width, canvas.height);
		makeColor(d.data);
		ctx.putImageData(d,0,0);
		
		layerBox.appendChild(document.createElement('br'));
		layerBox.style.textAlign="center";
		layerBox.append(layerName);
		//layerBox.innerText = layerName;
		layerBox.className = 'col-xs-3 nopadding';
		
		layersContainer.appendChild(layerBox);
	}
		
	predictionContainer.appendChild(layersContainer);
  
	const probsContainer = document.createElement('div');
	probsContainer.className="col-xs-2";
	
	if (recScore > 0.35){
		const row = document.createElement('div');
		row.className = 'row';
		row.textContent = "This image is too far out of our training distribution so we will not process it. (recScore:" + (Math.round(recScore * 100) / 100) + ")"
		probsContainer.appendChild(row);
	}else{
		const row = document.createElement('div');
		row.className = 'row';
		row.textContent = "Disease Predictions";
		row.style.fontWeight= "600";
		probsContainer.appendChild(row);
		
		for (let i = 0; i < classes.length; i++) {
		    const row = document.createElement('div');
		    row.className = 'row';
		    const classElement = document.createElement('div');
		    classElement.className = 'cell';
		    classElement.innerText = classes[i].className;
		    row.appendChild(classElement);
		    const probsElement = document.createElement('div');
		    probsElement.className = 'cell';
		    probsElement.innerText = (classes[i].probability.toFixed(2)*100) + "%";
		    scale = parseInt((1-classes[i].probability)*255)
		    probsElement.style.backgroundColor = "rgb(255," + scale + "," + scale + ")";
		    row.appendChild(probsElement);
		    probsContainer.appendChild(row);
		}
	}

	predictionContainer.appendChild(probsContainer);
	predictionsElement.insertBefore(document.createElement('hr'), predictionsElement.firstChild);
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

      img.onload = () => predict(img, f.name);
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

//run();
