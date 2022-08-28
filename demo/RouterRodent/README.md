# Router Rodent

## Available Scripts

In the project directory, you can run:

### Running on your computer

#### Start the development server

`npm run dev`

Runs the app in the development mode.<br>
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.<br>
You will also see any lint errors in the console.

#### Start the API server

`node ./server.js`

Runs the API server, so that the game can request traffic from netbeam

### Running it on Google Cloud

`npm run build`

Builds the app for production to the `build` folder.<br>
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.<br>
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

`gcloud app deploy`

Will deploy it to Google Cloud
