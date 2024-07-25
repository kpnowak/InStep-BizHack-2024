const express = require('express');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());

app.post('/place1', (req, res) => {
  console.log('Received at Place 1:', req.body);
  res.send('Data sent to Place 1');
});

app.post('/place2', (req, res) => {
  console.log('Received at Place 2:', req.body);
  res.send('Data sent to Place 2');
});

app.listen(3000, () => {
  console.log('Server is running on port 3000');
});
