//Package name ---

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

public class TreeCutDataRegister extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_tree_cut_data_register);

        getSupportActionBar().hide();

        String StateUT = getIntent().getStringExtra("StateUT");
        String District = getIntent().getStringExtra("District");


        NumberPicker treeCutNumber = findViewById(R.id.number1);



        treeCutNumber.setMinValue(0);
        treeCutNumber.setMaxValue(100);

        Button treesCutButton = findViewById(R.id.treescut);

        FirebaseDatabase database = FirebaseDatabase.getInstance();
        DatabaseReference treeRef = database.getReference().child("Trees")
                .child("Locations");


        treesCutButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (treeCutNumber.getValue() > 0) {

                    final String[] treeCutOriginalNumber = new String[1];
                    int numTreesCut = treeCutNumber.getValue();

                    treeRef.child(StateUT).child(District).child("TreesCut").addListenerForSingleValueEvent(new ValueEventListener() {
                        @Override
                        public void onDataChange(@NonNull DataSnapshot snapshot) {
                            treeCutOriginalNumber[0] = snapshot.getValue().toString();
                            int initial = Integer.parseInt(treeCutOriginalNumber[0]);

                            int value = initial + numTreesCut;

                            Intent intent = new Intent(TreeCutDataRegister.this, ValidateDataAndSubmit.class);
                            intent.putExtra("value", value);
                            intent.putExtra("StateUT", StateUT);
                            intent.putExtra("District", District);
                            intent.putExtra("cutorplant", "cut");
                            intent.putExtra("userTreeValue", numTreesCut);

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
