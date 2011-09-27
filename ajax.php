<?php 

if (isset($_POST['sendToValue'])){
	$value = $_POST['sendToValue'];	
}else{
	$value = "notavailable";
}
 
//echo json_encode(array("returnFromValue"=>"This is returned from PHP : ".$value));	


// // Parse without sections
// $ini_array = parse_ini_file("data/saffran.ini");
// echo json_encode($ini_array);

// Parse with sections
$ini_array = parse_ini_file("data/". $value . ".ini", true);
echo json_encode($ini_array);

?>