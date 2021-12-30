# Forest Tracker
## Introduction
Hey there! This app is all about protecting the environment! A unique method. Deforestation is happening at an alarming rate. We all know. Trees are also being cut illegally. 
So, this app is to check that.
The basic idea is to keep a track of every single tree cut or planted. So, we can exactly determine the percentage tree cover after the app usage has been started.
The first step to protecting the environment is to know our resources. 

## Description of the App
The app uses firebase to store data at backend.

-	App begins with a 3 page intro slider.
-	Next screen takes the user to simple login asking for name (not stored online, just for Hello!), and selecting Home state and district.
-	Main home screen has 3 bottom navigations and a “Upload” Floating action button

### Home Fragment
-   Greets user by “Hello {Name}”, on top
-   Contains 4 buttons
-   Report Planted Trees (for Home district)
-   Report Cut Trees (for Home district)
-   View All Data
-   View Your State Data
-   Also shows the summary of data uploaded by user.
    
### Track Fragment (for viewing data of a specific state)
-	Shows list of all states
 
### Dashboard Fragment
-	Shows Name, Home State and District
-	All Data Uploaded By User Button -> Takes to another screen and shows all data uploaded by user
-	Help Button -> Shows some help content
-	About Button -> Shows some about
-	Feedback / Suggest a Feature Button -> Gets some feedback from the user

### Upload Floating Action Button
- Activity 1: Get State and District from user.
- Activity 2: Buttons to report planted or cut trees. Also shows current data for the selected district. 
- Activity 3: Differs based on whether user selects to report planted or cut trees. Shows a number selector to pick a number. Then a Button to ‘Validate and Register’
- Activity 4: To check false data reports, user has to validate by using any of the three options
  -	Add an image proof
      -	User has to upload an image, that will be stored on firebase. 
  -	Enter tree id (a number marked on tree)
      -	User has to enter the number marked on tree
  -	Solve a simple math problem
      -	User solves a simple math problem

#### After validation, data is successfully uploaded.

## Firebase Data Structure:
~~~
forest-tracker
|    -Feedback
|	-Random UUID: “{Feedback}”
|
|
|    -Feature
|	-Random UUID: “{Suggested Feature}”
|
|
|    -Trees
|	-Locations
|	    -State 1
|		-District 1
|		...	
|		...
|	    -State 2
|		-District 2
|		...
|		...
|	    ...
|	    ...
|
|    -Math Problems
|	-State 1
|	    -District 1
|		- 'n' TreesPlanted DD-MM-YYYY_HH-MM-SS_AA: "Answered {Answer}"
|		- 'n' TreesCut DD-MM-YYYY_HH-MM-SS_AA: "Answered {Answer}"
|
|    -Tree Numbers
|	-State 1
|	    -District 1
|		- 'n' TreesPlanted DD-MM-YYYY_HH-MM-SS_AA: "{Tree ID}"
|		- 'n' TreesCut DD-MM-YYYY_HH-MM-SS_AA: "{Tree ID}"
~~~     

### Explanation
##### Database for locations has been maintained as shown above.
- Similarly, when someone uses “Math Problem” as validation, it is stored under “Math Problems” under State under District formatted as
    -	**‘n’ TreesPlanted DD-MM-YYYY_HH-MM-SS_AA: "Answered {Answer}"**
    -	**‘n’ TreesCut DD-MM-YYYY_HH-MM-SS_AA: "Answered {Answer}"**
-	When someone uses “Tree ID” as validation, it is stored under “Tree Numbers” under State under District formatted as
    -	**‘n’ TreesPlanted DD-MM-YYYY_HH-MM-SS_AA: {Tree ID} (as int)**
    -	**‘n’ TreesCut DD-MM-YYYY_HH-MM-SS_AA: {Tree ID} (as int)**
-	When someone uses Image Proof as validation, the image is uploaded to Firebase Storage under  \images\ by the name
    -	**{State}\_{District}\_’n’ Trees Planted_DD-MM-YYYY_HH-MM-SS_AA**
    -	**{State}\_{District}\_’n’ Trees Cut_DD-MM-YYYY_HH-MM-SS_AA**
-	When someone suggests a feature, it is stored under “Feature” under a random UUID as 
    -	**{UUID}: “Suggested Feature”**
-	When someone writes a feedback, it is stored under “Feedback” under a random UUID as 
    -	**{UUID}: “Feedback”**

## Privacy
- User’s privacy has been taken very seriously. You can see, instead of maintaining such a large database, location could be fetched easily using GPS, but it has not been done. 
- Similarly, the name is not uploaded anywhere. It is stored only in the phone only for greeting purposes. Same is the case with the Home state and district data.

## Future Developments:
1.	Add more validation options and optimize the existing ones. Like, someone can still enter false Tree ID or upload any image, there is no method to check that even. Math problem is also not apt here.
2.	Make the App’s UI and UX more interesting.
3.	Add some incentives. For instance, TreeCoins. A user would get, say, 10 TreeCoins for every tree cut/planted report. That would make it even more interesting. 
Also, can maintain a leaderboard and give some rewards, it would make people more willing to participate.

## Contribution
Any contribution on this project will be much appreciated.

### Thank You
