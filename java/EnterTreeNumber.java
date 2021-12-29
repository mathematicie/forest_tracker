package com.environment.foresttracker;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.View;
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

public class EnterTreeNumber extends AppCompatActivity {

    FirebaseDatabase database;
    DatabaseReference treeRef;

    Calendar calendar;
    String dateTime;
    SimpleDateFormat simpleDateFormat;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_enter_tree_number);

        database = FirebaseDatabase.getInstance();
        treeRef = database.getReference().child("Trees")
                .child("Locations");

        if (getSupportActionBar()!=null)
            getSupportActionBar().hide();

        int value = getIntent().getIntExtra("value", 0);
        String StateUT = getIntent().getStringExtra("StateUT");
        String District = getIntent().getStringExtra("District");
        String cutorplant = getIntent().getStringExtra("cutorplant");
        int userTreeValue = getIntent().getIntExtra("userTreeValue", 0);


        TextView showLocation, showDistrict, showData;
        Button uploadTreeNumber;
        TextInputLayout editText;

        editText = findViewById(R.id.editTextTreeNumber);


        uploadTreeNumber = findViewById(R.id.submitTreeNumber);



        showLocation = findViewById(R.id.showLocationOnTreeNumberActivity);
        showDistrict = findViewById(R.id.showDistrictOnTreeNumberActivity);
        showData = findViewById(R.id.showDataToUploadTreeNumberActivity);

        showLocation.setText(StateUT);


        showDistrict.setText(District);


        String toShow = "";
        if (cutorplant.equals("cut")) toShow = "Trees Cut:";
        else if (cutorplant.equals("plant")) toShow = "Trees Planted:";

        showData.setText("Data to upload:\n"+toShow+String.valueOf(userTreeValue));


        uploadTreeNumber.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (editText.getEditText().getText().toString().length()<4){
                    Toast.makeText(EnterTreeNumber.this, "Enter at least a 4 digit number.", Toast.LENGTH_SHORT).show();
                }
                else if (editText.getEditText().getText().toString().length()>=4){

                    calendar = Calendar.getInstance();
                    simpleDateFormat = new SimpleDateFormat("dd-MM-yyyy_HH-mm-ss_aaa");
                    dateTime = simpleDateFormat.format(calendar.getTime()).toString();



                    String finaltoShow = "";
                    if (cutorplant.equals("cut")) finaltoShow = "TreesCut";
                    else if (cutorplant.equals("plant")) finaltoShow = "TreesPlanted";


                    String treescutorplanted = "";
                    if (cutorplant.equals("cut")) treescutorplanted = "TreesCut";
                    else if (cutorplant.equals("plant")) treescutorplanted = "TreesPlanted";


                    int numberOfTree = Integer.parseInt(editText.getEditText().getText().toString());

                    database.getReference().child("Trees").child("Tree Numbers").child(StateUT).child(District)
                            .child(String.valueOf(userTreeValue)+" "+ finaltoShow+" "+dateTime).setValue(numberOfTree);

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


                    Toast.makeText(EnterTreeNumber.this, "Data Uploaded!", Toast.LENGTH_SHORT).show();

                    Intent intent = new Intent(EnterTreeNumber.this, MainActivity.class);
                    intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);

                    startActivity(intent);
                    finish();


                }
            }
        });


    }
}