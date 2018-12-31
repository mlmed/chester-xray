
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- The above 3 meta tags *must* come first in the head; any other head content must come *after* these tags -->
    <meta name="description" content="">
    <meta name="author" content="">
    <link rel="icon" href="../../favicon.ico">

    <title>Survival Prediction</title>


<!-- 	<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.3/jquery.min.js"></script> -->
	<script src="https://code.jquery.com/jquery-1.12.4.js"></script>
	<script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
	<link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

	<!-- Bootstrap core CSS -->
	
	<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css">
	<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap-theme.min.css">
	<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/js/bootstrap.min.js"></script>
   

<!-- 	<script language="javascript" type="text/javascript" src="js/flot/jquery.js"></script> -->
	<script language="javascript" type="text/javascript" src="lib/flot/jquery.flot.js"></script>
	<script language="javascript" type="text/javascript" src="lib/flot/jquery.flot.time.js"></script>
	
	
	<script src="lib/labeledslider/jquery.ui.labeledslider.js"></script>
	<link rel="stylesheet" href="lib/labeledslider/jquery.ui.labeledslider.css">

    <!-- Custom styles for this template
    <link href="navbar.css" rel="stylesheet"> -->

    <!-- Just for debugging purposes. Don't actually copy these 2 lines! -->
    <!--[if lt IE 9]><script src="../../assets/js/ie8-responsive-file-warning.js"></script><![endif]
    <script src="../../assets/js/ie-emulation-modes-warning.js"></script> -->

    <!-- HTML5 shim and Respond.js for IE8 support of HTML5 elements and media queries -->
    <!--[if lt IE 9]>
      <script src="https://oss.maxcdn.com/html5shiv/3.7.3/html5shiv.min.js"></script>
      <script src="https://oss.maxcdn.com/respond/1.4.2/respond.min.js"></script>
    <![endif]-->
  </head>

  <body>

    <div class="container">

      <!-- Static navbar -->
      <nav class="navbar navbar-default">
        <div class="container-fluid">
          <div class="navbar-header">
            <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#navbar" aria-expanded="false" aria-controls="navbar">
              <span class="sr-only">Toggle navigation</span>
              <span class="icon-bar"></span>
              <span class="icon-bar"></span>
              <span class="icon-bar"></span>
            </button>
            <a class="navbar-brand" href="#">Survival Prediction</a>
          </div>
          <div id="navbar" class="navbar-collapse collapse">
            <ul class="nav navbar-nav">
              <li class="active"><a href="#">Home</a></li>
              <li class="dropdown">
                <a href="#" class="dropdown-toggle" data-toggle="dropdown" role="button" aria-haspopup="true" aria-expanded="false">Dropdown <span class="caret"></span></a>
                <ul class="dropdown-menu">
                  <li><a href="#">Action</a></li>
                  <li><a href="#">Another action</a></li>
                  <li><a href="#">Something else here</a></li>
                  <li role="separator" class="divider"></li>
                  <li class="dropdown-header">Nav header</li>
                  <li><a href="#">Separated link</a></li>
                  <li><a href="#">One more separated link</a></li>
                </ul>
              </li>
            </ul>
            <ul class="nav navbar-nav navbar-right">
              <li><a href="#">About</a></li>
            </ul>
          </div><!--/.nav-collapse -->
        </div><!--/.container-fluid -->
      </nav>

<style>

.demo-container {
	box-sizing: border-box;
	padding: 20px 15px 15px 15px;
	margin: 0px auto 0px auto;
	border: 1px solid #ddd;
	background: #fff;
	background: linear-gradient(#f6f6f6 0, #fff 50px);
	background: -o-linear-gradient(#f6f6f6 0, #fff 50px);
	background: -ms-linear-gradient(#f6f6f6 0, #fff 50px);
	background: -moz-linear-gradient(#f6f6f6 0, #fff 50px);
	background: -webkit-linear-gradient(#f6f6f6 0, #fff 50px);
	box-shadow: 0 3px 10px rgba(0,0,0,0.15);
	-o-box-shadow: 0 3px 10px rgba(0,0,0,0.1);
	-ms-box-shadow: 0 3px 10px rgba(0,0,0,0.1);
	-moz-box-shadow: 0 3px 10px rgba(0,0,0,0.1);
	-webkit-box-shadow: 0 3px 10px rgba(0,0,0,0.1);
}
</style>



<script>

function softmax(arr) {
    return arr.map(function(value,index) { 
      return Math.exp(value) / arr.map( function(y /*value*/){ return Math.exp(y) } ).reduce( function(a,b){ return a+b })
    })
}


function drawPlot(){

	var numBins = 20

	var raw = new Array();
	for (var i = 0; i <= numBins; i++ ){
		number = Math.random()
		number += i*($( "#inputAge" ).slider("value")-$( "#inputDonorAge" ).slider("value"))/500

		if ($( "#inputBloodType" ).labeledslider("value") == "2"){
			number += (numBins-i)*2
		}
		
		raw.push(number);
	 }

	raw = softmax(raw)


	var pdf = new Array();
	for (var i = 0; i <= numBins; i++ ){
		pdf.push([i,raw[i]]);
	 }

	var cdf = new Array();

	for (var i = 0; i <= numBins; i++ ){
		if (i == 0)
			sum = 0
		else
			sum = raw.slice(0,i).reduce(function(acc, val) { return acc + val; });

		cdf.push([i,1-sum]);
	 }


	
		$.plot("#pdf", 
			[
				{ data:pdf, label:"&nbsp; PDF of Rejection", lines:{show:true}, points:{show:true}}
		    ], {
			xaxis: {
				tickSize: 1,
			},
	        yaxes: [{ 
	        	min:0,
	        	max:0.30,
	            axisLabel: "Prob. of Rejection",
	            axisLabelUseCanvas: true,
	            axisLabelFontSizePixels: 12,
	            axisLabelFontFamily: 'Verdana, Arial',
	            axisLabelPadding: 3,
	        }],
	        legend: {
	            noColumns: 0,
	            labelBoxBorderColor: "#000000",
	            position: "ne"
	        },
		});

		$.plot("#cdf", 
				[
					{ data:cdf, label:"&nbsp; Cumulative Survival Probability", lines:{show:true}, points:{show:true}}
			    ], {
				xaxis: {
					tickSize: 1,
				},
		        yaxes: [{ 
		        	min:0,
		        	max:1,
		            axisLabel: " Prob. of Rejection",
		            axisLabelUseCanvas: true,
		            axisLabelFontSizePixels: 12,
		            axisLabelFontFamily: 'Verdana, Arial',
		            axisLabelPadding: 3,
		        }],
		        legend: {
		            noColumns: 0,
		            labelBoxBorderColor: "#000000",
		            position: "ne"
		        },
			});
}


$(function() {

// 	$(".instantupdate").change(function(){
// 		console.log("a");
// 		drawPlot();
// 	});


	$( "#inputAge" ).slider({
	      value:50,
	      min: 0,
	      max: 100,
	      step: 1,
	    slide: function(event,ui){
	    	$(this).find(".ui-slider-handle").text( ui.value );
	    	drawPlot();  
	    },
    	create: function() {
    		$(this).find(".ui-slider-handle").text( $( this ).slider( "value" ) );
      }
	});


	
    var handle = $( "#inputDonorAge .ui-slider-handle" );
	$( "#inputDonorAge" ).slider({
	      value:50,
	      min: 0,
	      max: 100,
	      step: 1,
		    slide: function(event,ui){
		    	$(this).find(".ui-slider-handle").text( ui.value );
		    	drawPlot();  
		    },
	    	create: function() {
	    		$(this).find(".ui-slider-handle").text( $( this ).slider( "value" ) );
	      }
	});



	
    var handle = $( "#inputBloodType .ui-slider-handle" );
    values = ["A","B", "AB", "O"]
	$( "#inputBloodType" ).labeledslider({
	      value:0,
	      min: 0,
	      max: 3,
	      step: 1,
// 		    slide: function(event,ui){
// 		    	$(this).find(".ui-slider-handle").text( values[ui.value] );
// 		    	drawPlot();  
// 		    },
// 	    	create: function() {
// 	    		$(this).find(".ui-slider-handle").text( values[$( this ).labeledslider( "value" )] );
// 	      }
	});

    $( "#inputBloodType" ).labeledslider( 'option', 'tickLabels', ["A","B", "AB", "O"] )
	
	
	drawPlot();

	var t=setInterval(drawPlot,100);
});


	</script>


  <style>
  .ui-slider-handle-local {
    width: 3em !important; 
    height: 1.6em !important;
    top: 50% !important;
    margin-top: -.8em;
    margin-left: -1.5em !important;
    text-align: center;
    line-height: 1.6em;
  }
  </style>
<div class="row">
	<div class="col-md-6">
	
		Input features of upload a spreadsheet with attributes <input type="file">
		
		<br><br>
		
		

	
		<div class="form-group">
		  <label class="control-label" for="inputAge">Patient Age</label>
		  <div class="slidecontainer">
		  		<div  id="inputAge" class="slider"><div class="ui-slider-handle ui-slider-handle-local"></div></div>
		  </div>
		</div>
		
		<div class="form-group">
		  <label class="control-label" for="inputDonorAge">Donor Age</label>
		  <div class="slidecontainer">
		  		<div  id="inputDonorAge" class="slider"><div class="ui-slider-handle ui-slider-handle-local"></div></div>
		  </div>
		</div>
		
		<div class="form-group">
		  <label class="control-label" for="inputBloodType">Donor Blood Type</label>
		  <div class="slidecontainer">
		  		<div  id="inputBloodType" class="slider"><div class="ui-slider-handle ui-slider-handle-local"></div></div>
		  </div>
		</div>
		
		
	</div>
	
	
	
	<div class="col-md-6">

		<div style="font: 18px/1.5em 'proxima-nova', Helvetica, Arial, sans-serif;">
			<div class="demo-container" style="width:100%;height:220px">
				<div id="pdf" class="" style="	width: 100%; height: 100%;font-size: 14px;"></div>
			</div>
		</div>
		
		<div style="font: 18px/1.5em 'proxima-nova', Helvetica, Arial, sans-serif;">
			<div class="demo-container" style="width:100%;height:220px">
				<div id="cdf" class="" style="	width: 100%; height: 100%;font-size: 14px;"></div>
			</div>
		</div>

	</div>
</div>








    </div> <!-- /container -->
  </body>
</html>
