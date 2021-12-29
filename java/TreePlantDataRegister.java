package com.environment.foresttracker;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.NumberPicker;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

public class TreePlantDataRegister extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_tree_plant_data_register);

        getSupportActionBar().hide();

        String StateUT = getIntent().getStringExtra("StateUT");
        String District = getIntent().getStringExtra("District");


        NumberPicker treePlantNumber = findViewById(R.id.number2);

        treePlantNumber.setMinValue(0);
        treePlantNumber.setMaxValue(100);

        Button treesPlantedButton = findViewById(R.id.treesplanted);


        FirebaseDatabase database = FirebaseDatabase.getInstance();
        DatabaseReference treeRef = database.getReference().child("Trees")
                .child("Locations");

        treesPlantedButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (treePlantNumber.getValue() > 0) {
                    int numTreesPlanted = treePlantNumber.getValue();
                    final String[] treePlantOriginalNumber = new String[1];

                    treeRef.child(StateUT).child(District).child("TreesPlanted").addListenerForSingleValueEvent(new ValueEventListener() {
                        @Override
                        public void onDataChange(@NonNull DataSnapshot snapshot) {
                            treePlantOriginalNumber[0] = snapshot.getValue().toString();
                            int initial = Integer.parseInt(treePlantOriginalNumber[0]);

                            int value = initial + numTreesPlanted;


                            Intent intent = new Intent(TreePlantDataRegister.this, ValidateDataAndSubmit.class);
                            intent.putExtra("value", value);
                            intent.putExtra("StateUT", StateUT);
                            intent.putExtra("District", District);
                            intent.putExtra("cutorplant", "plant");
                            intent.putExtra("userTreeValue", numTreesPlanted);


                            startActivity(intent);



                        }

                        @Override
                        public void onCancelled(@NonNull DatabaseError error) {

                        }
                    });

                }
            }
        });



    }
}