//Package name ---

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.os.Bundle;
import android.widget.TextView;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.util.ArrayList;

public class Districts extends AppCompatActivity {

    String stateUT;
    ArrayList<TreeClass> list;
    DistrictDisplayAdapter adapter;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_districts);


        if (getSupportActionBar()!=null)
        getSupportActionBar().hide();

        list = new ArrayList<>();

        stateUT = getIntent().getStringExtra("StateUT");

        FirebaseDatabase database = FirebaseDatabase.getInstance();
        DatabaseReference treeRef = database.getReference().child("Trees")
                .child("Locations");

        TextView showStateUT;

        showStateUT = findViewById(R.id.showStateUTOnDistrictsList);
        showStateUT.setText(stateUT);

        treeRef.child(stateUT).addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot2) {
                for(DataSnapshot dataSnapshot2: snapshot2.getChildren()) {
                    String LocalDistrict = dataSnapshot2.getKey().toString();



                    String treescut = dataSnapshot2.child("TreesCut").getValue().toString();
                    String treesplanted = dataSnapshot2.child("TreesPlanted").getValue().toString();

                    list.add(new TreeClass(stateUT, LocalDistrict, treescut, treesplanted));

                    adapter.notifyDataSetChanged();


                }

            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {
            }
        });


        RecyclerView listView = findViewById(R.id.listShow);

        adapter = new DistrictDisplayAdapter(list);
        listView.setLayoutManager(new LinearLayoutManager(this));
        listView.setAdapter(adapter);





    }
}
