for i in {1..399}; do
  for suffix in 0 1; do
    wget "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/CXR_png/MCUCXR_$(printf "%04d" $i)_$suffix.png"
  done
done

mv *.png dataset/

##categorize the downloaded dataset

# Define the source folder where your images are located.
source_folder="dataset"

# Create the destination folders if they don't exist.
mkdir -p "${source_folder}/0"
mkdir -p "${source_folder}/1"

# Iterate through the files in the source folder.
for file in "${source_folder}"/*; do
    # Check if the file ends with "_0.png".
    if [[ "${file}" =~ _0\.png$ ]]; then
        mv "${file}" "${source_folder}/0"
        echo "Moved ${file} to 0"
    # Check if the file ends with "_1.png".
    elif [[ "${file}" =~ _1\.png$ ]]; then
        mv "${file}" "${source_folder}/1"
        echo "Moved ${file} to 1"
    fi
done
 



