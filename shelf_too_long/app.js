const express = require('express');
const bodyParser = require('body-parser');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');

const app = express();
const port = 3001;

// Middleware to parse JSON bodies
app.use(bodyParser.json());

// Endpoint to receive product details and forecasted discount
    // Read the JSON file
    fs.readFile('data.json', 'utf8', (err, data) => {
        if (err) {
            console.error(`Error reading JSON file: ${err}`);
            return res.status(500).send({ error: 'Internal Server Error' });
        }

        try {
            // Parse the JSON data
            const jsonData = JSON.parse(data);
            const { "Product name": productName, "Manufacture date": manufactureDate, Quantity } = jsonData[0];

            if (!productName || !Quantity || !manufactureDate) {
                return res.status(400).send({ error: 'Missing required fields' });
            }

            // Calculate the expiry_time as the number of days since manufacture date
            const expiryTime = Math.floor((new Date() - new Date(manufactureDate)) / (1000 * 60 * 60 * 24));

            // Construct the command to run the Python script
            const command = `python forecast.py ${expiryTime} ${Quantity} "${productName}"`;

            // Execute the Python script
            exec(command, (error, stdout, stderr) => {
                if (error) {
                    console.error(`exec error: ${error}`);
                    return res.status(500).send({ error: 'Internal Server Error' });
                }
                if (stderr) {
                    console.error(`stderr: ${stderr}`);
                    return res.status(500).send({ error: 'Internal Server Error' });
                }

                console.log(`Python script output: ${stdout.trim()}`);

                // Read the forecasted discount from the JSON file
                fs.readFile('forecasted_discount.json', 'utf8', (err, forecastData) => {
                    if (err) {
                        console.error(`Error reading JSON file: ${err}`);
                        return res.status(500).send({ error: 'Internal Server Error kurwa' });
                    }
            
                    
                        const jsonData1 = JSON.parse(forecastData);
                        const { forecasted_discount, product_type } = jsonData1;
            
                        if (forecasted_discount === undefined || product_type === undefined) {
                            return res.status(400).send({ error: 'Missing required fields' });
                        }
            
                        console.log(`Product Type: ${product_type}`);
                        console.log(`Forecasted Discount: ${forecasted_discount}`);
                });
            });
        } catch (err) {
            console.error(`Error parsing JSON data: ${err}`);
            return res.status(500).send({ error: 'Internal Server Error' });
        }
    });

app.post('/survey', (req, res) => {
    let { hobbies, groceries } = req.body;
  
    // Log the received data
    console.log('Received Survey Data:');
    console.log('Hobbies:', hobbies);
    console.log('Groceries:', groceries);
    console.log('Hobbies Type:', typeof hobbies);
    console.log('Groceries Type:', typeof groceries);
  
    // Convert hobbies and groceries to arrays if they are strings
    if (typeof hobbies === 'string') {
        hobbies = hobbies.split(',').map(item => item.trim());
    }
    if (typeof groceries === 'string') {
        groceries = groceries.split(',').map(item => item.trim());
    }
  
    // If the fields are empty strings, convert them to empty arrays
    hobbies = hobbies.filter(item => item);
    groceries = groceries.filter(item => item);
  
    // Create an object to store the data
    const surveyData = {
        hobbies,
        groceries
    };
  
    // Define the path to the file
    const filePath = path.join(__dirname, 'survey_data.json');
  
    // Write JSON data to a file
    fs.writeFile(filePath, JSON.stringify(surveyData, null, 2), (err) => {
        if (err) {
            console.error('Error writing to file', err);
            return res.status(500).send('Error saving survey data');
        } else {
            console.log(`File saved successfully to ${filePath}`);
            res.status(200).send('Survey data received and saved to JSON file.');
        }
    });
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});