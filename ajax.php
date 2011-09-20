<?php 

if (isset($_POST['sendToValue'])){
	$value = $_POST['sendToValue'];	
}else{
	$value = "";
}
 
echo json_encode(array("returnFromValue"=>"This is returned from PHP : ".$value));	

?>