//Package name ---

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.Observer;
import androidx.lifecycle.ViewModelProvider;

import com.environment.foresttracker.R;
import com.environment.foresttracker.databinding.FragmentDashboardBinding;
import com.environment.foresttracker.databinding.FragmentHomeBinding;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.ArrayList;

public class HomeFragment extends Fragment {


    View rootView;
    TextView showName;


    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {

        rootView = inflater.inflate(R.layout.fragment_home, container, false);


        Context context = getActivity();

        if (((AppCompatActivity)getActivity()).getSupportActionBar() != null )
            ((AppCompatActivity)getActivity()).getSupportActionBar().hide();


        SharedPreferences sharedPreferences = getActivity().getSharedPreferences("details", Context.MODE_PRIVATE);

        String getName = sharedPreferences.getString("user_name", "");
        String StateUT = sharedPreferences.getString("stateUT", "");
        String District = sharedPreferences.getString("district", "");

        showName = rootView.findViewById(R.id.showNameHome);
        showName.setText(getName);


        ArrayList<SaveTreeClass> list;

        Gson gson = new Gson();
        String json = sharedPreferences.getString("allUserData", null);
        Type type = new TypeToken<ArrayList<SaveTreeClass>>(){}.getType();

        list = gson.fromJson(json, type);
        int totalTrees=0;
        int planted=0, cut=0;

        if(list==null){
            totalTrees=0;
            planted=0;
            cut=0;
        }
        else if (list != null){
        for(int i=0; i<list.size(); i++){
            totalTrees = totalTrees+ Integer.parseInt(list.get(i).treenumber);

            String cutorplant = (list.get(i).cutOrPlant);
            if(cutorplant.equals("cut")) {
                cut = cut+ Integer.parseInt(list.get(i).treenumber);
            }
            else if (cutorplant.equals("plant")){
                planted = planted+ Integer.parseInt(list.get(i).treenumber);
            }
        }
        }

        TextView treesSummary;
        treesSummary = rootView.findViewById(R.id.showSummary);

        treesSummary.setText("Tree Planted Reports: " + String.valueOf(planted) + "\n"
                + "Tree Cut Reports: " + String.valueOf(cut));



        Button reportPlantedTrees, reportCutTrees, viewAllData, viewYourStateData;

        reportCutTrees = rootView.findViewById(R.id.reportCutTrees);
        reportPlantedTrees = rootView.findViewById(R.id.reportPlantedTrees);
        viewAllData = rootView.findViewById(R.id.viewAllData);
        viewYourStateData = rootView.findViewById(R.id.viewYourStateData);

        reportCutTrees.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getContext(), TreeCutDataRegister.class);
                intent.putExtra("StateUT", StateUT);
                intent.putExtra("District", District);
                startActivity(intent);

            }
        });

        reportPlantedTrees.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(getContext(), TreePlantDataRegister.class);
                intent.putExtra("StateUT", StateUT);
                intent.putExtra("District", District);
                startActivity(intent);

            }
        });
        viewAllData.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(getContext(), TrackAllData.class));

            }
        });

        viewYourStateData.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent intent = new Intent(getContext(), Districts.class);
                intent.putExtra("StateUT", StateUT);
                startActivity(intent);
            }
        });



        return rootView;
    }



}
