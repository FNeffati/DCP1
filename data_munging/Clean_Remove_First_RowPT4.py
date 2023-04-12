import csv

# open the input CSV file
with open('Updated_Combo.csv', newline='') as input_file:

    # read the CSV data into a list
    data = list(csv.reader(input_file))

    # separate the header row
    header = data[0]

    # remove the first data row
    data_without_header = data[2:]

    # open the output CSV file
    with open('Updated_Combo_new.csv', 'w', newline='') as output_file:

        # create a CSV writer object
        writer = csv.writer(output_file)

        # write the header row to the output file
        writer.writerow(header)

        # write the modified data to the output file
        writer.writerows(data_without_header)
