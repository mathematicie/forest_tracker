//Package name ---

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

public class ShowSingleStateAndData extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_show_single_state_and_data);

        getSupportActionBar().hide();

        TextView showStateUT = findViewById(R.id.showLocation);
        TextView showDistrict = findViewById(R.id.showDistrict);

        String StateUT = getIntent().getStringExtra("StateUT");
        String District = getIntent().getStringExtra("District");


        showStateUT.setText(StateUT);
        showDistrict.setText(District);

        FirebaseDatabase database = FirebaseDatabase.getInstance();
        DatabaseReference treeRef = database.getReference().child("Trees")
                .child("Locations");


        TextView showNetTrees = findViewById(R.id.showNetTrees);

        Button addTreeCut = findViewById(R.id.addTreesCutData);
        Button addTreePlanted = findViewById(R.id.addTreesPlantedData);

        addTreeCut.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(ShowSingleStateAndData.this, TreeCutDataRegister.class);
                intent.putExtra("StateUT", StateUT);
                intent.putExtra("District", District);
                startActivity(intent);
            }
        });

        addTreePlanted.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(ShowSingleStateAndData.this, TreePlantDataRegister.class);
                intent.putExtra("StateUT", StateUT);
                intent.putExtra("District", District);
                startActivity(intent);
            }
        });









        treeRef.child(StateUT).child(District).addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                String treeplant = snapshot.child("TreesPlanted").getValue().toString();
                String treecut = snapshot.child("TreesCut").getValue().toString();

                int treeplant2 = Integer.parseInt(treeplant);
                int treecut2 = Integer.parseInt(treecut);
                String toShow = "";
                if (treeplant2>treecut2) {
                    int net = treeplant2-treecut2;
                    toShow = "Gain of "+String.valueOf(net)+" trees.";
                }
                else if (treecut2>treeplant2) {
                    int net = treecut2-treeplant2;
                    toShow = "Loss of "+String.valueOf(net)+" trees.";
                }
                else if (treecut2==treeplant2){
                    toShow = "No net gain or loss.";
                }
                showNetTrees.setText("Since App Usage Started:\n" +
                        "Trees Cut: "+treecut+"\n" +
                        "Trees Planted: " +treeplant+"\n" +
                        "\n" +
                        toShow);
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {

            }
        });



    }
}
