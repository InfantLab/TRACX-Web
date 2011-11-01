/***********
 * TRACX Javascript Calculator
 * 
 * Implements the Truncated Recursive Atoassociative Chunk eXtractor 
 * (TRACX, French, Addyman & Mareschal, Psych Rev, 2011) A neural network
 * that performs sequence segmentation and chunk extraction in artifical
 * grammar learning tasks and statistical learning tasks.
 *
 * Note it uses the sylvester maths libaries to perform matrix multiplications
 *  http://sylvester.jcoglan.com/
 */
var TRACX = (function () { 
    var API = {}; //a variable to hold public interface for this module
    API.Version = '0.1.1'; //version number for 
    API.VersionDate = "31-September-2011";
    
    //private variables
    var trainingData,userEncodings, inputEncodings,
   		weightsInputToHidden, weightsHiddenToOutput, //weight matrices
   		OLD_deltaWeightsInputToHidden, OLD_deltaWeightsHiddenToOutput,  //old matrices for momentum calc
   		testWords, testPartWords, testNonWords, //test items
    	trackingFlag, trackingBigrams, trackingInterval, trackingSteps, trackingResults; //tracking Learning
   	
    //default parameters
    var params = {
    	learningRate: 0.04,
    	recognitionCriterion: 0.4,
		reinforcementThreshold: 0.25,
		momentum: 0,
		temperature: 1.0,
		fahlmanOffset: 0.1,
		bias: -1,
		sentenceRepetitions: 1,
		numberSubjects: 1
	};
    
    //variables for stting through
	var currentStep, maxSteps,startSimulation, net, testResults;
     
    API.setParameters = function (parameters) { 
        params = parameters;
    }; 
    API.getParameters = function () { 
        return params;
    };
    API.getWeightsIn2Hid = function () { 
        return weightsInputToHidden;
    };
    API.getWeightsHid2Out = function () { 
        return weightsHiddenToOutput;
    };
    API.setTrainingData = function (TrainingData) {
		trainingData = TrainingData;
    };
    API.setTestData = function (testData) {
		testWords=testData.Words;
		testPartWords=testData.PartWords;
		testNonWords=testData.NonWords;
    };
    API.getTestData = function () {
		return {Words:testWords,PartWords:testPartWords,NonWords:testNonWords};
    };
   	API.setTrackingData = function(data){
   		trackingFlag = data.flag;
   		//have to tidy up the bigrams strings a bit
   		trackingBigrams = [];
   		var temp = data.bigrams.split(',');
   		for (x in temp){
   			if (temp[x].trim().length===2){
				trackingBigrams[temp[x].trim()] = temp[x].trim();
			}
		}
   		trackingInterval = data.interval;
   	}
   	API.getTrackingData = function(){
		return {flag:trackingFlag,bigrams:trackingBigrams.join(','),interval:trackingInterval};
   	}
   	API.getCurrentStep = function(){
   		return currentStep;
   	}
   	//function to reset the step counter so training is restarted
   	API.reset = function(){
   		currentStep  = -1;
   		testResults = null;
   	}
   	API.getInputEncoding = function(letter){
   		return inputEncodings[letter];	
   	}
   
    function isNumber(n) {
	  return !isNaN(parseFloat(n)) && isFinite(n);
	}

   	//find the unique elements of array - useful for getting all possible phonemes/syllables
	function unique(array) {
	    var o = {}, i, l = array.length, r = [];
	    for(i=0; i<l;i+=1) o[array[i]] = array[i];
	    for(i in o) r.push(o[i]);
	    return r;
	}
	//create an array of length len filled with with value val
	function newFilledArray(len, val) {
	    var array = [];
	    for (var i = 0; i < len; i++) {
	        array[i] = val;
	    }
	    return array;
	}
	//length function for object arrays
	Object.size = function(obj) {
	    var size = 0, key;
	    for (key in obj) {
	        if (obj.hasOwnProperty(key)) size++;
	    }
	    return size;
	};

	
	API.getInputEncodings = function(){
		if (userEncodings){
			return inputEncodings;
		}else if (!trainingData){
			return null;
		}
		//generate the input vectors
		inputEncodings = [];
		var letters = unique(trainingData); //find unique letters
		var zeroArray = newFilledArray(letters.length,-1);
		for(var i=0, l=letters.length;i<l;i++){
			//each input encoded as zeros everywhere
			var thisInput = zeroArray.slice(0); //make a new copy of array
			//except for i-th dimension
			thisInput[i]=1;
			inputEncodings[letters[i]]=thisInput;
		}
		console.log(inputEncodings);
		return inputEncodings;			
	};
	
	/***
	 * create appropriately sized starting Weight matrices
	 */
	API.initializeWeights = function(){
		if (!inputEncodings){
			return null;
		}
		var N = Object.size(inputEncodings); // Get the size of input vector
		//console.log(N);
		if (N<2){
			return null;
		}
		//initalise to small values around -.1 to +.1
		var temp = Matrix.Random(2*N+1,N).multiply(-1);
		weightsInputToHidden = Matrix.Random(2*N+1,N).add(temp);
		var temp2 = Matrix.Random(N+1,2*N).multiply(-1);
		weightsHiddenToOutput = Matrix.Random(N+1,2*N).add(temp2);
       //remove matrices for momemtum weights too
        OLD_deltaWeightsInputToHidden = null;
        OLD_deltaWeightsHiddenToOutput = null;
	};
	
	//sum up the elements of array
	function sum (x) {
		for(var i=0,sum=0;i<x.length;sum+=x[i++]);
		return sum;
	}
	//mean value of elements in array
	function mean (x) {
	    return sum(x) / x.length;
	}
	//standard deviation function.
	function stdev(x) {
		var variance = 0.0;
		var n = x.length;
		var v1 = 0.0;
		var v2 = 0.0;
		var stddev =0.0;
		var meanx = mean(x);
		if (n != 1)	{
			for (var i = 0; i <= n - 1; i++)
			{
				v1 = v1 + (x[i] - meanx) * (x[i] - meanx);
				v2 = v2 + (x[i] - meanx);
			}
			v2 = v2*v2 / n;
			variance = (v1 - v2) / (n-1);
			if (variance < 0) { variance = 0; }
			stddev = Math.sqrt(variance);
		}
		return stddev;
	}
	
	function tanh (x) {
    	//slow tanh fn
    	// sinh(number)/cosh(number)
	    return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
	}
	function rational_tanh(x){
		//fast approx tanh fn
		//http://stackoverflow.com/questions/6118028/fast-hyperbolic-tangent-approximation-in-javascript
	    if(x<-3)
	        return -1;
	    else if(x>3)
	        return 1;
	    else
	        return x*(27 + x*x)/(27 + 9*x*x);
	}
	function tanh_deriv(x){
		//derivative of the -1 to 1 tanh squashing fcn
		return  params.temperature * (1 - x*x) + params.fahlmanOffset;
	}

	API.ActivationFn = function(array){
		for(x in array){
			array[x] = rational_tanh(array[x]);	
		}
	};
	/***
	 * feed forward through the network
	 */
	API.networkOutput = function(Input1,Input2){
		 //build input & treat as vector so we can right-multiply
		var InputVector = $M(Input1).transpose().augment($M(Input2).transpose());
		// }
       //add on the bias
		var InputVectorBias = InputVector.augment($V([params.bias])); 
		//multiply by first weight matrix
		var Hid_net_in_acts =  InputVectorBias.x(weightsInputToHidden);
		//pass through activation fn
		//var Hid_out_acts = Hid_net_in_acts.map(rational_tanh);
		var Hid_out_acts = Hid_net_in_acts.map(tanh);
		Hid_out_acts = Hid_out_acts.augment($V([params.bias])); //add bias node
		//multiply by second weight matrix
		var Output_net_in_acts = Hid_out_acts.x(weightsHiddenToOutput);
		//pass through activation fn
		//var Output_out_acts = Output_net_in_acts.map(rational_tanh);
		var Output_out_acts = Output_net_in_acts.map(tanh);
		
		//calculate the delta between input and output
		var del = Output_out_acts.subtract(InputVector);
		del = del.map(Math.abs);
		return {In:InputVector,Hid: Hid_out_acts, Out: Output_out_acts, Delta:del.max()};
	};
  	
  	function backPropogateError(net){
  		//TODO 
  		//This code has not been optimised in any way.
  		
  		//we want output to be same as input
  		//so error to backProp is the difference between input and output
		var Errors_RAW = net.Out.subtract(net.In);
		//so output errors is each diff multiplied by appropriate 
		//derivative of output activation
		//1st get deriv
		var dOut = net.Out.map(tanh_deriv);
		
		var Errors_OUTPUT = []; 
		var dim = Errors_RAW.dimensions();
		for(var c=1; c<dim.cols+1;c++){
			Errors_OUTPUT.push(Errors_RAW.e(1,c) * dOut.e(1,c));
		}
		Errors_OUTPUT = $M(Errors_OUTPUT);
		
		//So change weights is this deriv times hidden activations
		var dE_dw = net.Hid.transpose().x(Errors_OUTPUT.transpose());
		//multiplied by learning rate and with momentum added
		var HO_dwts;
		if (OLD_deltaWeightsHiddenToOutput){
			HO_dwts = dE_dw.x(-1 * params.learningRate).add(OLD_deltaWeightsHiddenToOutput.x(params.momentum));
		}else{
			//haven't got any old deltas yet
			HO_dwts = dE_dw.x(-1 * params.learningRate);
		}
		//update 2nd layer weights
		weightsHiddenToOutput = weightsHiddenToOutput.add(HO_dwts);
		//copy old delta for momentum calc.
		OLD_deltaWeightsHiddenToOutput = HO_dwts.dup();
		
		//Errors on hidden layer are oouput errors back propogated
		var Errors_HIDDEN_RAW = Errors_OUTPUT.transpose().x(weightsHiddenToOutput.transpose());
		//again multiplied by appropriated activations
		var dHidOut = net.Hid.map(tanh_deriv);
		var Errors_HIDDEN = [];
		dim = Errors_HIDDEN_RAW.dimensions();
		//Errors_Hidden less bias node so skip the last col
		for(var c=1; c<dim.cols;c++){
			Errors_HIDDEN.push(Errors_HIDDEN_RAW.e(1,c) * dHidOut.e(1,c));
		}
		Errors_HIDDEN = $M(Errors_HIDDEN);
		
		dE_dw = net.In.augment($V([params.bias])).transpose().x(Errors_HIDDEN.transpose());  // no IH_dwts associated with the bias
		var IH_dwts;
		if (OLD_deltaWeightsInputToHidden){
			IH_dwts = dE_dw.x(-1*params.learningRate).add(OLD_deltaWeightsInputToHidden.x(params.momentum));
		}else{
			IH_dwts = dE_dw.x(-1*params.learningRate);
		}
		//update 1st layer weights
		weightsInputToHidden = weightsInputToHidden.add(IH_dwts);
		OLD_deltaWeightsInputToHidden = IH_dwts.dup();
  	}
  
    API.trainNetwork = function (steps, progressCallback) { 
	    /*********************/
	    try{
	        console.log("trainNetwork");
			//how many steps do we train for on this call
			var untilStep;
			if (steps < 0){
				untilStep = maxSteps;
			}else{
				untilStep = Math.min(maxSteps,currentStep+steps);
			}
			//
			// the main training loop
	        while (currentStep< untilStep){
				currentStep++;
				// progressCallback(1,'.');
				//read and encode the first bit of training data
				var Input_t1, Input_t2;	
				if (net.Delta < params.recognitionCriterion){
					//new input is hidden unit representation
					Input_t1 = net.Hid.elements[0];
					//not including bias
					Input_t1.length = Input_t1.length -1;
				}else{
					//input is next training item
					Input_t1 = inputEncodings[trainingData[i]];
				}
				Input_t2 = inputEncodings[trainingData[i+1]];

				net = API.networkOutput(Input_t1,Input_t2);
				
				// if on input the LHS comes from an internal representation then only
				// do a learning pass 25% of the time, since internal representations,
				// since internal representations are attentionally weaker than input
				// from the real, external world.
				if ((net.Delta > params.recognitionCriterion) //train netowrk if error large
				 || (Math.random() <= params.reinforcementThreshold))//or if small error and below threshold
				 {
					backPropogateError(net);
					net = API.networkOutput(Input_t1,Input_t2);
				}
				if (trackingFlag && currentStep%trackingInterval == 1){
					trackingSteps.push(currentStep);
					//if tracking turned on we test the network 
					//at fixed intervals with a set of test bigrams
					for(x in trackingBigrams){
						var ret = API.testString(trackingBigrams[x]);
						trackingResults[trackingBigrams[x]].push([currentStep, ret.meanDelta]);
					}
				}
			}
			return true;
	    }
	    catch(err){
	    	console.log(err);
	    	progressCallback(1,"TRACX.trainNetwork Err: " + err.message + "<br/>");
	    	return false;
	    }
    }; 
    
    /***
     * a function to test what the network has learned.
     * pass a comma seperated list of test words, it
     * tests each one returning deltas and mean delta
     */
    API.testStrings = function (testStrings) { 
		var testString = testStrings.split(",");
		var deltas = [];
		var toterr = 0;
		var wordcount = 0;
		for(w in testString){
			if (testString[w].length>1){
		    	ret = TRACX.testString(testString[w]);
		    	toterr += ret.totalDelta;
		    	wordcount++; 
		    	deltas.push(toterr.toFixed(3));
			}
		}
		if (wordcount > 0){
			return {delta:deltas,meanDelta:toterr/wordcount};
		}else{
			return {delta:null,meanDelta:null};
		}
		
	};
    /***
     * a function to test what the network has learned.
     * returns the total delta error on each letter pair in a string
     * and a total delta for the word.
     */
    API.testString = function (inString) { 
		try{
			var len = inString.length;
	  		var net = {Delta: 500};
	  		var delta = [];
	  		var totDelta = 0;
	  		for(var i=0; i<len-1;i++){
	  			if (net.Delta < params.recognitionCriterion){
					//new input is hidden unit representation
					Input_t1 = net.Hidnet.Hid.elements[0];
				}else{
					//input is next training item
					Input_t1 = inputEncodings[inString[i]];
				}
				Input_t2 = inputEncodings[inString[i+1]];
				net = API.networkOutput(Input_t1,Input_t2);
				delta.push(net.Delta);
				totDelta += net.Delta;
			}
	  		return {deltas:			delta,
	  				totalDelta:		totDelta, 
	  				meanDelta:	totDelta/(len-1),
	  				activations:net};
	  	}
	    /*******************/
	    catch(err){
	    	console.log(err);
			return -1000;
	    }  		
  	};
  	
  	/******
  	 * The function which gets called when someone presses the calculate button.
  	 */
  	API.runFullSimulation = function(progressCallback){
  		
  		startSimulation = new Date();
  		currentStep = 0;
  		maxSteps = params.sentenceRepetitions * (trainingData.length -1);
  		if (progressCallback){
  			progressCallback(1,"Simulation started: " + startSimulation.toLocaleTimeString() + "<br/>");
  		}
  		
  		//set up the object to store results
  		var testResults = {	trainSuccess:	false,
  						   	elapsedTime:	-1,
  						   	Words:			{mean:-1,sd:-1,all:[]},
  							PartWords:		{mean:-1,sd:-1,all:[]},
  							NonWords:		{mean:-1,sd:-1,all:[]},
  							trackingSteps:	null,
  							trackingOutputs:null};
        if (trackingFlag){
    		//initialise stacked array to store tracking data
    		trackingResults = [];
    		trackingSteps = [];
        	for(x in trackingBigrams){
    			trackingResults[trackingBigrams[x].trim()] = [];
	        }
	    }

		progressCallback('Subjects: ');
		//loop round with a new network each time
	 	for (var run_no=0; run_no<params.numberSubjects;run_no++){
			currentStep == 0;
  			net = {Delta: 500};  // some very big delta to start with
	  		API.initializeWeights();
			if (progressCallback){
  				progressCallback(1,(1 + run_no) + ",");
				progressCallback(0,"Run: " + (1 + run_no) + "<br/>");
				progressCallback(0,'Initial Weight Matrices<br/>Input to Hidden<br/>');
				progressCallback(0, weightsInputToHidden.inspect());
				progressCallback(0,'<br/>Hidden to Output<br/>');
				progressCallback(0,weightsHiddenToOutput.inspect());
				
			}
			
			if(API.trainNetwork(-1, progressCallback)){
				//training worked for this subject
				testResults.trainSuccess =true;
				/////////////////////////////////// ////
				//TESTING THE NETWORK
				var	ret = TRACX.testStrings(testWords );
				if (ret.meanDelta){			
			    	testResults.Words.all.push(ret.meanDelta);
			   	}
				ret =  TRACX.testStrings(testPartWords );
		    	if (ret.meanDelta){			
			    	testResults.PartWords.all.push(ret.meanDelta);			        
		       	}
				ret =  TRACX.testStrings(testNonWords );
		    	if (ret.meanDelta){			
			    	testResults.NonWords.all.push(ret.meanDelta);			        
		       	}		    	
	       	}		
	    }
	    if (testResults.Words.all.length>0){
			testResults.Words.mean = mean(testResults.Words.all);
	    	testResults.Words.sd = stdev(testResults.Words.all);
	    }
	    if (testResults.PartWords.all.length>0){
			testResults.PartWords.mean = mean(testResults.PartWords.all);
	    	testResults.PartWords.sd = stdev(testResults.PartWords.all);
	    }
	    if (testResults.NonWords.all.length>0){
			testResults.NonWords.mean = mean(testResults.NonWords.all);
	    	testResults.NonWords.sd = stdev(testResults.NonWords.all);
	    }
	    var end = new Date();
	    testResults.elapsedTime = (end.getTime() - startSimulation.getTime())/1000;
	    if (progressCallback){
  			progressCallback(1,"Simulation finished: " + end.toLocaleTimeString() + "<br/>");
  			progressCallback(1,"Duration: " + testResults.elapsedTime.toFixed(3) + " secs.<br/>");
  		}
	    if (trackingFlag){
	    	testResults.trackingSteps = trackingSteps;
	    	testResults.trackingOutputs = trackingResults;
	    }
       	return testResults;   	
  	};
	
	
	/******
  	 * The function which will step through the training process so user can see
  	 * what is going on.
  	 */
  	API.stepThroughTraining = function(stepSize, progressCallback){
  		if (!currentStep || currentStep < 0){
  			//initialize things
  			startSimulation = new Date();
	  		if (progressCallback){
	  			progressCallback(1,"Simulation started: " + startSimulation.toLocaleTimeString() + "<br/>");
	  		}
	  		currentStep = 0;
  			net = {Delta: 500};  // some very big delta to start with
	  		API.initializeWeights();
			maxSteps = params.sentenceRepetitions * (trainingData.length -1);
  			if (progressCallback){
  				progressCallback(1, "Stepping through once,");
				progressCallback(0,"Stepping through once <br/>");
				progressCallback(0,'Weight Matrices<br/>Input to Hidden<br/>');
				progressCallback(0, weightsInputToHidden.inspect());
				progressCallback(0,'<br/>Hidden to Output<br/>');
				progressCallback(0,weightsHiddenToOutput.inspect());
			}	
	  		//set up the object to store results
	  		testResults = {	trainSuccess:	false,
  						   	elapsedTime:	-1,
  						   	Words:			{mean:-1,sd:-1,all:[]},
  							PartWords:		{mean:-1,sd:-1,all:[]},
  							NonWords:		{mean:-1,sd:-1,all:[]},
  							trackingSteps:	null,
  							trackingOutputs:null};
	        if (trackingFlag){
				//initialise stacked array to store tracking data
	    		trackingResults = [];
	    		trackingSteps = [];
	        	for(x in trackingBigrams){
	    			trackingResults[trackingBigrams[x].trim()] = [];
		        }
		    }
  		}
  		//convert percentage based step sizes
  		if (!isNumber(stepSize) && stepSize.indexOf("%")>0){
  			stepSize= 0.01 * maxSteps * parseFloat(stepSize);
  		}
  		
		// progressCallback(1, 'Subjects: ');
		// //loop round with a new network each time
	 	// for (var run_no=0; run_no<params.numberSubjects;run_no++){
			if(API.trainNetwork(stepSize, progressCallback)){
				//training worked for this subject
				testResults.trainSuccess =true;
				/////////////////////////////////// ////
				//TESTING THE NETWORK
				var	ret = TRACX.testStrings(testWords );
				if (ret.meanDelta){			
			    	testResults.Words.all.push(ret.meanDelta);
			   	}
				ret =  TRACX.testStrings(testPartWords );
		    	if (ret.meanDelta){			
			    	testResults.PartWords.all.push(ret.meanDelta);			        
		       	}
				ret =  TRACX.testStrings(testNonWords );
		    	if (ret.meanDelta){			
			    	testResults.NonWords.all.push(ret.meanDelta);			        
		       	}		    	
	       	}		
	    // }
	    if (testResults.Words.all.length>0){
			testResults.Words.mean = mean(testResults.Words.all);
	    	testResults.Words.sd = stdev(testResults.Words.all);
	    }
	    if (testResults.PartWords.all.length>0){
			testResults.PartWords.mean = mean(testResults.PartWords.all);
	    	testResults.PartWords.sd = stdev(testResults.PartWords.all);
	    }
	    if (testResults.NonWords.all.length>0){
			testResults.NonWords.mean = mean(testResults.NonWords.all);
	    	testResults.NonWords.sd = stdev(testResults.NonWords.all);
	    }
	    var end = new Date();
	    testResults.elapsedTime = (end.getTime() - startSimulation.getTime())/1000;
	    if (progressCallback){
  			progressCallback(1,"Simulation finished: " + end.toLocaleTimeString() + "<br/>");
  			progressCallback(1,"Duration: " + testResults.elapsedTime.toFixed(3) + " secs.<br/>");
  		}
	    if (trackingFlag){
	    	testResults.trackingSteps = trackingSteps;
	    	testResults.trackingOutputs = trackingResults;
	    }
       	return testResults;   	
  	};


    return API; //makes the tracx methods available  
}());
