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

prog = "\\";

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

const RECSCORE_THRESH = 0.5;
const OODSCORE_THRESH = 1000;
const GUI_WAITTIME = 10;

//const MODEL_PATH = 'https://mlmed.github.io/tools/xray/models/chestxnet-45rot15trans15scale4byte';
const MODEL_PATH = './models/xrv-all-45rot15trans15scale';
//const AEMODEL_PATH = 'https://mlmed.github.io/tools/xray/models/ae-chest-savedmodel-64-512';
//const AEMODEL_PATH = './models/ae-chest-savedmodel-64-512';

let chesternet;
let aechesternet;
let catElement;
let filesElement; 
let predictionsElement;
//let MODEL_CONFIG;

$(function(){

	if (findGetParameter("randomorder") == "true"){
		$("#info").text($("#info").text() + " In random order mode");
	}

	$.ajax({
		url: MODEL_PATH + "/config.json",
		dataType: "json",
		async:false,
		cache:false,
		error:function(jqXHR, textStatus, errorThrown){
			console.log(jqXHR);
			console.log(textStatus + errorThrown);
		},
		success: function(obj) {
			window.MODEL_CONFIG = obj;
		}
	});


	filesElement = document.getElementById('files');
	filesElement.addEventListener('change', async evt => {
		let files = evt.target.files;

		idxs = [...Array(files.length).keys()]
		if (findGetParameter("randomorder") == "true"){
			console.log("In random order mode");
			idxs.sort(() => Math.random() - 0.5);
		}
		for (var i = 0; i < idxs.length; i++) {
			f = files[idxs[i]]

			// Only process image files (skip non image files)
			if (!f.type.match('image.*')) {
				return;
			}

			let reader = new FileReader();
			//const idx = i;

			var deferred = $.Deferred();

			reader.onload = e => {
				let img = document.createElement('img');
				img.src = e.target.result;

				img.onload = async g => {
					console.log("Processing " + f.name);
					await predict(img, false, f.name);
					deferred.resolve();
				}
			};
			reader.readAsDataURL(f);

			await deferred.promise();
			
		}
		$("#files").val("");
	});

	predictionsElement = document.getElementById('predictions');
		
});

async function run(){

	try{
		await load_model();
		//await load_model_debug();
		
		await run_demo();
		
		document.getElementById('file-container').style.display = '';
		
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

realfetch = window.fetch
cachedfetch = function(arg) {
	console.log("Forcing cached version of " + arg)
    return realfetch(arg, {cache: "force-cache"})
}

let downloadStatus
async function load_model(){
	status('Loading model...');
	console.log("load_model");
	
	const startTime = performance.now();
	downloadStatus=setInterval(display_size_data,100);
	window.fetch = cachedfetch
	
	try{
		chesternet = await tf.loadGraphModel('indexeddb://' + MODEL_PATH);
	}catch{
		chesternet = await tf.loadGraphModel(MODEL_PATH + "/model.json", fetchFunc=cachedfetch);
		
	}
	try{
		await chesternet.save('indexeddb://' + MODEL_PATH);
	}catch{
		// try to clean up the local caches
		const dbs = await window.indexedDB.databases()
		dbs.forEach(db => { window.indexedDB.deleteDatabase(db.name) })
	}
	
	console.log("First Model loaded " + Math.floor(performance.now() - startTime) + "ms");
	if (typeof AEMODEL_PATH !== 'undefined'){
		aechesternet = await tf.loadGraphModel(AEMODEL_PATH + "/model.json", fetchFunc=cachedfetch);
		console.log("Second Model loaded " + Math.floor(performance.now() - startTime) + "ms");
	}
	window.fetch = realfetch
	clearInterval(downloadStatus);

	status('Loading model into memory...');

	await sleep(GUI_WAITTIME)

	//chesternet.predict(tf.zeros([1, 1, MODEL_CONFIG.IMAGE_SIZE, MODEL_CONFIG.IMAGE_SIZE])).dispose();
	
	if (typeof AEMODEL_PATH !== 'undefined'){
		aechesternet.predict(tf.zeros([1, 1, 64, 64])).dispose();
	}
	status('');
};

// this is just to debug the UI without waiting for the real model to load
async function load_model_debug(){
	chesternet = {};
	chesternet.predict = (img) => {
		return tf.zeros([1, 18]).add(0.3);
	};
	chesternet.execute = (img) => {
		return tf.zeros([1, 18]).add(0.3);
	};
}
	

async function run_demo(){
	
	console.log("run_demo");	
	
	var imgElement = new Image();
	imgElement.onload = () => {
		predict(imgElement, true, "Example Image (" + imgElement.src.substring(imgElement.src.lastIndexOf('/')+1)+ ")");
		};
	imgElement.src = "examples/auntminnie-d-2020_01_28_23_51_6665_2020_01_28_Vietnam_coronavirus.jpeg";	
	
}


//let batched;
let aebatched;
let currentpred;
async function predict(imgElement, isInitialRun, name) {

	try{
		$("#file-container #files").attr("disabled", true)
		$(".computegrads").each((k,v) => {v.style.display = "none"});

		const startTime = performance.now();
		await predict_real(imgElement, isInitialRun, name);

		$(".loading").each((k,v) => {v.style.display = "none"});
		const totalTime = performance.now() - startTime;
		status(`Done in ${Math.floor(totalTime)}ms`);

	}catch(err) {
		$(".loading").hide()
		status("Error! " + err.message);
		console.log(err)

		if (err.name == "BadBrowser"){
			$("#file-container #files").attr("disabled", true);
		}
	}
	$("#file-container #files").attr("disabled", false)
}

function prepare_image_resize_crop(imgElement, size){
	
	orig_width = imgElement.width
	orig_height = imgElement.height
	if (orig_width < orig_height){
		imgElement.width = size
		imgElement.height = Math.floor(size*orig_height/orig_width)
	}else{
		imgElement.height = size
		imgElement.width = Math.floor(size*orig_width/orig_height)
	}
	
	console.log("img wxh: " + orig_width + ", " + orig_height + " => " + imgElement.width + ", " + imgElement.height)	
	
	img = tf.browser.fromPixels(imgElement).toFloat();
	
	hOffset = Math.floor(img.shape[1]/2 - size/2)
	wOffset = Math.floor(img.shape[0]/2 - size/2)
	
	img_cropped = img.slice([wOffset,hOffset],[size,size])
	
	img_normalized = img_cropped.div(tf.scalar(255)).mul(tf.scalar(MODEL_CONFIG.IMAGE_SCALE));
	meanImg = img_normalized.mean(2);
	
	return meanImg
}

function prepare_image(currentpred, imgElement){
	
	currentpred[0].img_original = tf.browser.fromPixels(imgElement).toFloat();

	currentpred[0].img_highres = prepare_image_resize_crop(imgElement, Math.max(imgElement.width, imgElement.height));
	
	currentpred[0].img_input = prepare_image_resize_crop(imgElement, MODEL_CONFIG.IMAGE_SIZE);
	
}

async function predict_real(imgElement, isInitialRun, name) {
	status('Predicting...');
	console.log("predict_real");
	
	const startTime = performance.now();
		
	currentpred = $("#predtemplate").clone();
	currentpred.find(".loading").each((k,v) => {v.style.display = "block"});
	currentpred[0].id = name
	currentpred[0].grads = []; // to cache grads
	predictionsElement.insertBefore(currentpred[0], predictionsElement.firstChild);
	
	$(".btn-reset-layers").click(function(){
		
		currentpred.find(".gradimage").hide();
		reset_grad_btns(currentpred);
	})
	
	$(".btn-invert-colors").click(function(){
		
		if (currentpred.find(".inputimage_highres").css("filter") == "invert(1)"){
			currentpred.find(".inputimage_highres").css("filter", "invert(0)");
			$(".btn-invert-colors").removeClass("active");
		}else{
			currentpred.find(".inputimage_highres").css("filter", "invert(1)");
			$(".btn-invert-colors").addClass("active");
		}
	})

	//currentpred.find(".inputimage").attr("src", imgElement.src)
	currentpred.show();

	currentpred.find(".imagename").text(name)
	
	prepare_image(currentpred, imgElement);

	//////// display input image
	img = currentpred.find(".inputimage_highres")
	await tf.browser.toPixels(currentpred[0].img_highres.div(tf.scalar(MODEL_CONFIG.IMAGE_SCALE)),img[0]);	
	currentpred.find(".inputimage_highres").show()
/*	img = currentpred.find(".inputimage")
	await tf.browser.toPixels(currentpred[0].img_input.div(tf.scalar(MODEL_CONFIG.IMAGE_SCALE)),img[0]);	
	currentpred.find(".inputimage").show()*/
	await sleep(GUI_WAITTIME)
	////////////////////

	console.log("Prepared input image " + Math.floor(performance.now() - startTime) + "ms");

	if (typeof AEMODEL_PATH !== 'undefined'){
		
		//	status('Computing Reconstruction...');
	
		img_small = document.createElement('img');
		img_small.src = imgElement.src
		img_small.width = 64
		img_small.height = 64
		
		let {recInput, recErr, rec} = tf.tidy(() => {
	
			const img = tf.browser.fromPixels(img_small).toFloat();
	
			const normalized = img.div(tf.scalar(255));
	
			aebatched = normalized.mean(2).reshape([1, 1, 64, 64])
	
			//const batched2 = batched.mul(2).sub(1)
	
			const rec = aechesternet.predict(aebatched)
			console.log(rec);
	
			const recErr = aebatched.sub(rec).abs()
	
			return {recInput:aebatched, recErr: recErr, rec: rec};
		});
	
		recScore = recErr.mean().dataSync()
		console.log("recScore" + recScore);
		console.log("Computed Reconstruction " + Math.floor(performance.now() - startTime) + "ms");
		if (isInitialRun && (recScore > 0.27 || recScore < 0.01)){
			error = new Error("Something wrong with this browser. Try refreshing the page. (" + recScore + ")");
			error.name="BadBrowser"
			throw error
		}

		canvas_a = currentpred.find(".inputimage_rec")[0]
		layer = recInput.reshape([64,64])
		await tf.browser.toPixels(layer.div(2).add(0.5),canvas_a);
	
		canvas_b = currentpred.find(".recimage")[0]
		layer = rec.reshape([64,64])
		await tf.browser.toPixels(layer.div(2).add(0.5),canvas_b);
	
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
		await tf.browser.toPixels(layer.clipByValue(0, 1),canvas);
/*		canvas.style.width = "100%";
		canvas.style.height = "";
		canvas.style.imageRendering = "pixelated";*/
	
		ctx = canvas.getContext("2d");
		d = ctx.getImageData(0, 0, canvas.width, canvas.height);
		makeColor(d.data);
		makeTransparent(d.data)
		ctx.putImageData(d,0,0);
	
		scoreBox = document.createElement("center")
		score = "recScore:" + parseFloat(recScore).toFixed(2)  + ", ssim:" + ssim.ssim.toFixed(2) 
		scoreBox.innerText = score
		currentpred.find(".oodimagebox")[0].append(scoreBox)
	
		//currentpred.find(".oodviz .loading")[0].style.display = "none";
		currentpred.find(".oodimagebox")[0].style.display = "block";
		currentpred.find(".oodtoggle")[0].style.display = "block";
	
		currentpred.find(".oodtoggle").click(function(){$(this).closest(".prediction").find(".oodimage").toggle()});
	
	
		can_predict = ssim.ssim > 0.60
		
		//	////////////////////
	
		console.log("Plotted Reconstruction " + Math.floor(performance.now() - startTime) + "ms");
	}else{
		recScore = 0.01;
		can_predict = true;
	}
	
	status('Predicting disease...');

	if (!can_predict){

		showProbError(currentpred.find(".predbox")[0], score)
		return

	}else{
		output = tf.tidy(() => {

			const batched = currentpred[0].img_input.reshape([1, 1, MODEL_CONFIG.IMAGE_SIZE, MODEL_CONFIG.IMAGE_SIZE])
			return chesternet.execute(batched, [MODEL_CONFIG.OUTPUT_NODE])
		});
		
		await sleep(GUI_WAITTIME)
		
		logits = await output.data()

		console.log("Computed logits and grad " + Math.floor(performance.now() - startTime) + "ms");
		console.log("logits=" + logits)


		currentpred[0].logits = logits;
		currentpred[0].classes = await distOverClasses(logits);
		
		showProbResults(currentpred)//, logits, recScore)
		currentpred.find(".predviz .loading").hide();
		currentpred.find(".loading").hide();
		//currentpred.find(".computegrads").show();


		currentpred.find(".oodtoggle").hide();
		console.log("results plotted " + Math.floor(performance.now() - startTime) + "ms");
	}

}

function reset_grad_btns(thispred){
	
	thispred.find(".explain-btn").removeClass("active");
	thispred.find(".explain-btn").removeClass("btn-primary");
	thispred.find(".explain-btn").addClass("btn-info");
}

async function computeGrads(thispred, idx, explainElement){

	try{
		status('Computing gradients...' + idx + " " + MODEL_CONFIG.LABELS[idx]);

		//thispred.find(".computegrads").hide();
		//thispred.find(".gradimagebox").hide();
		if ($(explainElement).hasClass("active")){
			$(explainElement).addClass("btn-info");
			$(explainElement).removeClass("btn-primary");
			$(explainElement).removeClass("active");
			thispred.find(".gradimage").hide();
			return
		}else{
			reset_grad_btns(thispred);
			
			$(explainElement).removeClass("btn-info");
			$(explainElement).addClass("btn-primary");
			$(explainElement).addClass("active");
		}

		thispred.find(".desc").text("");

		$("#file-container #files").attr("disabled", true)

		const startTime = performance.now();
		await computeGrads_real(thispred, idx);

		const totalTime = performance.now() - startTime;
		status(`Done in ${Math.floor(totalTime)}ms`);

	}catch(err) {
		$(".loading").hide()
		status("Error! " + err.message);
		console.log(err)
	}

	$("#file-container #files").attr("disabled", false)
}

async function computeGrads_real(thispred, idx){

	//batched = thispred[0].batched
	thispred.find(".viewbox .loading").show();
	//await sleep(1000)

	//cache computation
	if (thispred[0].grads[idx] == undefined){
		await sleep(GUI_WAITTIME)
	
		//saveasdasd = await chestgrad.save('indexeddb://' + MODEL_PATH + "-chestgrad");
		
		//chestgrad = await tf.loadGraphModel('indexeddb://' + MODEL_PATH + "-chestgrad");
	
		layer = tf.tidy(() => {
	
			chestgrad = tf.grad(x => chesternet.predict(x).reshape([-1]).gather(idx))
			
			const batched = thispred[0].img_input.reshape([1, 1, MODEL_CONFIG.IMAGE_SIZE, MODEL_CONFIG.IMAGE_SIZE])
			const grad = chestgrad(batched);
	
			const layer = grad.mean(0).abs().max(0)
			return layer.div(layer.max())
	
		});
		
		//////// display grad image
		canvas = thispred.find(".gradimage")[0]
		await tf.browser.toPixels(layer,canvas);	
	
	    ctx = canvas.getContext("2d");
		d = ctx.getImageData(0, 0, canvas.width, canvas.height);
		makeColor(d.data);
		makeTransparent(d.data)
	
	    thispred[0].grads[idx] = d
    }

    d = thispred[0].grads[idx]

	ctx = canvas.getContext("2d");
	ctx.putImageData(d,0,0);
	thispred.find(".gradimage").show()

	thispred.find(".viewbox .loading").hide()
	//thispred.find(".gradimagebox").show()
	thispred.find(".desc").text("Predictive regions for " + MODEL_CONFIG.LABELS[idx])

}

async function distOverClasses(values){

	const topClassesAndProbs = [];
	for (let i = 0; i < values.length; i++) {

		if (values[i]<MODEL_CONFIG.OP_POINT[i]){
			value_normalized = values[i]/(MODEL_CONFIG.OP_POINT[i]*2);
		}else{
			value_normalized = 1-((1-values[i])/((1-(MODEL_CONFIG.OP_POINT[i]))*2));
			if (value_normalized>0.6 & MODEL_CONFIG.SCALE_UPPER){
			value_normalized = Math.min(1, value_normalized*MODEL_CONFIG.SCALE_UPPER);
			}
		}
		console.log(MODEL_CONFIG.LABELS[i] + ",pred:" + values[i] + "," + "OP_POINT:" + MODEL_CONFIG.OP_POINT[i] + "->normalized:" + value_normalized);

		topClassesAndProbs.push({
			className: MODEL_CONFIG.LABELS[i],
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
//		var r = pix[i],
		g = pix[i+1];
//		b = pix[i+2];

		if (g < 20){ 
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

function showProbResults(currentpred) {

	classes = currentpred[0].classes
	predictionContainer = currentpred.find(".predbox")[0]

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
			classElement.style.whiteSpace="nowrap";
			classElement.style.overflowX="scroll";
			classElement.style.maxWidth="150px";
			probsElement.className = 'cell gradient';
			probsElement.style.borderBottomStyle="solid";
			probsElement.style.borderColor = "white";
			probsElement.style.borderWidth="2";
		}
		row.appendChild(classElement);

		const explainElement = document.createElement('button');
		if (i == -1 || classes[i].probability <= 0.6){
			explainElement.style.visibility = "hidden"
		}else{
			explainElement.className = 'explain-btn btn btn-info';
			explainElement.innerText = "explain";
			$(explainElement).click(function(){computeGrads(currentpred,i,explainElement)})	    	
		}
		row.appendChild(explainElement);



		probsElement.style.width="100%";
		probsElement.style.textAlign="left";
		probsElement.style.position="relative";

		if (i == -1){
			target = document.createElement('span');
			target.innerText = "Healthy";
			target.style.position="absolute";
			target.style.fontSize="x-small";
			target.style.minWidth="100px"
			target.style.left=0;
			probsElement.appendChild(target)

			target = document.createElement('span');
			target.innerText = "Risk";
			target.style.position="absolute";
			target.style.fontSize="x-small";
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

//			target = document.createElement('span');
//			//target.innerText = "|";
//			target.className="glyphicon glyphicon-menu-right"
//			target.style.marginLeft="-7px"; //glyh is 14x14
//			target.style.position="absolute";
//			target.style.left=parseInt(currentpred[0].PPV80[i].probability*100) + "%";
//			target.style.fontWeight="900";
//			probsElement.appendChild(target)

//			target = document.createElement('span');
//			//target.innerText = "|";
//			target.className="glyphicon glyphicon-menu-left"
//			target.style.marginLeft="-7px"; //glyh is 14x14
//			target.style.position="absolute";
//			target.style.left=parseInt(currentpred[0].NPV95[i].probability*100) + "%";
//			target.style.fontWeight="900";
//			probsElement.appendChild(target)
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
//				if (parseInt(a.style.left)>60){
//				a.style.marginLeft="-30px";
//				}
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
		await tf.browser.toPixels(layer.div(layer.max()),canvas);		
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
			classElement.onClick = computeGrads(thispred, batched, [i]);
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



function sleep(ms) {
	return new Promise(resolve => setTimeout(resolve, ms));
}

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

function downloadCSV(){
	download("chester-export.csv",computeCSV());
}

function computeCSV(){
	lines = []
	e = $(".prediction")[0]
	line = "Filename"
		$(e.classes).each(function(k,l){line += ("," + l.className)})
		lines += (line + "\n")
		$(".prediction").each(function(i,e){
			if (e.id != "predtemplate"){
				line = e.id
				$(e.classes).each(function(k,l){line += ("," + l.probability)})
				lines += (line + "\n")
			}
		})
		console.log(lines);
	return lines
}

//https://stackoverflow.com/questions/3665115/how-to-create-a-file-in-memory-for-user-to-download-but-not-through-server
function download(filename, text) {
	var element = document.createElement('a');
	element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
	element.setAttribute('download', filename);

	element.style.display = 'none';
	document.body.appendChild(element);

	element.click();

	document.body.removeChild(element);
}

//run();
