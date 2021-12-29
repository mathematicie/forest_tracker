//Package name ---

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.environment.foresttracker.databinding.FragmentTrackBinding;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.util.ArrayList;

public class TrackFragment extends Fragment {




    ArrayList<String> stateUTList;
    StateUTListAdapter adapter;
    View rootView;

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {

        rootView = inflater.inflate(R.layout.fragment_track, container, false);

        ((AppCompatActivity)getActivity()).getSupportActionBar().hide();


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



        RecyclerView listView = rootView.findViewById(R.id.recyclerViewStateUT);
        adapter = new StateUTListAdapter(stateUTList);
        listView.setLayoutManager(new LinearLayoutManager(getContext()));
        listView.setAdapter(adapter);



        return rootView;
    }


}
