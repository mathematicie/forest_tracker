package com.environment.foresttracker;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.os.Bundle;
import android.view.View;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.util.ArrayList;

public class TrackAllData extends AppCompatActivity {

    ArrayList<String> stateUTList;
    StateUTListAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_track_all_data);

        getSupportActionBar().hide();

        stateUTList = new ArrayList<>();


        FirebaseDatabase database = FirebaseDatabase.getInstance();
        DatabaseReference treeRef = database.getReference().child("Trees")
                .child("Locations");

        treeRef.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                for (DataSnapshot dataSnapshot: snapshot.getChildren()) {
                    String StateUT = dataSnapshot.getKey().toString();

                    stateUTList.add(StateUT);
                    adapter.notifyDataSetChanged();



                }
            }
            @Override
            public void onCancelled(@NonNull DatabaseError error) {

            }
        });



        RecyclerView listView = findViewById(R.id.recyclerViewStateUTActivity);
        adapter = new StateUTListAdapter(stateUTList);
        listView.setLayoutManager(new LinearLayoutManager(this));
        listView.setAdapter(adapter);




    }
}