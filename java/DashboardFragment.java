package com.environment.foresttracker;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.fragment.app.Fragment;

public class DashboardFragment extends Fragment {

    View rootView;


    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {

        rootView = inflater.inflate(R.layout.fragment_dashboard, container, false);


         if (((AppCompatActivity)getActivity()).getSupportActionBar() != null)
             ((AppCompatActivity)getActivity()).getSupportActionBar().hide();

        TextView name, stateUT, district;
        Button seeAllData, help, about, feedback;



        name = rootView.findViewById(R.id.showNameDashboard);
        stateUT = rootView.findViewById(R.id.showStateUTDashboard);
        district = rootView.findViewById(R.id.showDistrictDashboard);

        feedback = rootView.findViewById(R.id.feedback);

        feedback.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(getContext(), Feedback.class));
            }
        });

        seeAllData = rootView.findViewById(R.id.seeAllDataUploaded);
        help = rootView.findViewById(R.id.help);
        about = rootView.findViewById(R.id.about);

        SharedPreferences sharedPreferences = getContext().getSharedPreferences("details", Context.MODE_PRIVATE);


        String userName, userStateUT, userDistrict;

        userName = sharedPreferences.getString("user_name","");
        userStateUT = sharedPreferences.getString("stateUT","");
        userDistrict = sharedPreferences.getString("district","");

        name.setText(userName);
        stateUT.setText(userStateUT);
        district.setText(userDistrict);

        seeAllData.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(getContext(), SeeAllDataByUser.class));
            }
        });

        help.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(getContext(), Help.class));
            }
        });

        about.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(getContext(), About.class));

            }
        });

        return rootView;
    }


}