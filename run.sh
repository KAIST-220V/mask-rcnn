COMMAND=$1

while true; do
	$COMMAND
	
	if [ $? -eq 0 ]; then
		echo "Command succeeded with exit code 0."
		break
	else
		echo "Command failed with exit code $?. Retrying..."
	fi

done
