package com.environment.foresttracker;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.View;
import android.webkit.WebView;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.material.textfield.TextInputLayout;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Random;

public class SolveMath extends AppCompatActivity {

    Random random = new Random();
    int answer=0;

    String equation="";

    FirebaseDatabase database;
    DatabaseReference treeRef;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_solve_math);

        database = FirebaseDatabase.getInstance();
        treeRef = database.getReference().child("Trees")
                .child("Locations");


        getSupportActionBar().hide();


        int value = getIntent().getIntExtra("value", 0);
        String StateUT = getIntent().getStringExtra("StateUT");
        String District = getIntent().getStringExtra("District");
        String cutorplant = getIntent().getStringExtra("cutorplant");
        int userTreeValue = getIntent().getIntExtra("userTreeValue", 0);


        TextView showMathProblem,  showLocation, showDistrict, showData;
        Button submit;
        TextInputLayout input;

        submit = findViewById(R.id.submitMath);

        input = findViewById(R.id.editTextMath);

        showMathProblem = findViewById(R.id.showMathProblem);
        showData = findViewById(R.id.showDataToUploadMathActivity);
        showLocation = findViewById(R.id.showLocationOnMathActivity);
        showDistrict = findViewById(R.id.showDistrictOnMathActivity);

        showLocation.setText(StateUT);
        showDistrict.setText(District);

        String toShow = "";
        if (cutorplant.equals("cut")) toShow = "Trees Cut: ";
        else if (cutorplant.equals("plant")) toShow = "Trees Planted: ";

        showData.setText("Data to upload:\n"+toShow+String.valueOf(userTreeValue));

        getEquation();




        showMathProblem.setText("Math Problem:\n"+equation);


        submit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                if (!input.getEditText().getText().toString().equals(String.valueOf(answer))){
                    Toast.makeText(SolveMath.this, "Enter correct answer", Toast.LENGTH_SHORT).show();
                }
                else if (input.getEditText().getText().toString().equals(String.valueOf(answer))){

                    Calendar calendar;
                    String dateTime;
                    SimpleDateFormat simpleDateFormat;


                    calendar = Calendar.getInstance();
                    simpleDateFormat = new SimpleDateFormat("dd-MM-yyyy_HH-mm-ss_aaa");
                    dateTime = simpleDateFormat.format(calendar.getTime()).toString();



                    String finaltoShow = "";
                    if (cutorplant.equals("cut")) finaltoShow = "TreesCut";
                    else if (cutorplant.equals("plant")) finaltoShow = "TreesPlanted";


                    String treescutorplanted = "";
                    if (cutorplant.equals("cut")) treescutorplanted = "TreesCut";
                    else if (cutorplant.equals("plant")) treescutorplanted = "TreesPlanted";

                    database.getReference().child("Trees").child("Math Problems").child(StateUT).child(District)
                            .child(String.valueOf(userTreeValue)+" "+ finaltoShow+" "+dateTime).setValue("Answered: "+String.valueOf(answer));


                    treeRef.child(StateUT).child(District).child(treescutorplanted).setValue(value);

                    SharedPreferences sharedPreferences = getSharedPreferences("details", MODE_PRIVATE);
                    SharedPreferences.Editor editor = sharedPreferences.edit();

                    Gson getGson = new Gson();
                    String getJson = sharedPreferences.getString("allUserData", null);
                    Type type = new TypeToken<ArrayList<SaveTreeClass>>(){}.getType();


                    ArrayList<SaveTreeClass> list;
                    list = new ArrayList<>();


                    list = getGson.fromJson(getJson, type);

                    if (list != null){
                        list.add(new SaveTreeClass(StateUT, District, String.valueOf(userTreeValue), cutorplant, dateTime.replace("_", " ")));
                    }
                    else {
                        list = new ArrayList<>();
                        list.add(new SaveTreeClass(StateUT, District, String.valueOf(userTreeValue), cutorplant, dateTime.replace("_", " ")));
                    }


                    Gson gson = new Gson();

                    String json = gson.toJson(list);
                    editor.putString("allUserData", json);
                    editor.commit();


                    Toast.makeText(SolveMath.this, "Data Uploaded!", Toast.LENGTH_SHORT).show();

                    Intent intent = new Intent(SolveMath.this, MainActivity.class);
                    intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);

                    startActivity(intent);
                    finish();


                }
            }


        });



    }




    public int getRandomNumbers(){
        return  random.nextInt(15)+1;
    }

    public void getEquation(){

        int num1 = getRandomNumbers();
        int num2 = getRandomNumbers();
        int num3 = getRandomNumbers();
        int num4 = getRandomNumbers();

        char operator1 = '*';
        char operator2 = '+';
        char operator3 = '-';

        equation = ""+num1+operator1+num2+operator2+num3+operator3+num4;

        int number = num1;
        number = num1*num2;
        number = number+num3;
        number = number-num4;
        answer = number;

    }






}