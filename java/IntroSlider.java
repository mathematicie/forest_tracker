package com.environment.foresttracker;

import androidx.appcompat.app.AppCompatActivity;
import androidx.viewpager.widget.ViewPager;

import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;

public class IntroSlider extends AppCompatActivity {

    ViewPager viewPager;
    ImageView next, circle1, circle2, circle3;

    List<Integer> list;

    ViewPagerAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_intro_slider);


        SharedPreferences sharedPreferences = getSharedPreferences("details", MODE_PRIVATE);
        SharedPreferences.Editor editor = sharedPreferences.edit();

        boolean first = sharedPreferences.getBoolean("firstLaunch", true);
        if (!first){
            startActivity(new Intent(IntroSlider.this, Login.class));
            finish();
        }

        viewPager = findViewById(R.id.viewPager);
        next = findViewById(R.id.nextSlide);

        circle1 = findViewById(R.id.circle1);
        circle2 = findViewById(R.id.circle2);
        circle3 = findViewById(R.id.circle3);



        if (getSupportActionBar()!=null)
            getSupportActionBar().hide();

        list = new ArrayList<>();

        adapter = new ViewPagerAdapter(IntroSlider.this, list);

        viewPager.setAdapter(adapter);


        list.add(R.layout.welcome_page1);
        list.add(R.layout.welcome_page2);
        list.add(R.layout.welcome_page3);


        adapter.notifyDataSetChanged();


        viewPager.setCurrentItem(0);
        circle1.setColorFilter((getResources().getColor(R.color.circle_active)));
        circle2.setColorFilter((getResources().getColor(R.color.circle_inactive)));
        circle3.setColorFilter((getResources().getColor(R.color.circle_inactive)));


        next.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                int current = viewPager.getCurrentItem();

                if(current==0){
                    viewPager.setCurrentItem(1);
                    circle2.setColorFilter((getResources().getColor(R.color.circle_active)));
                    circle1.setColorFilter((getResources().getColor(R.color.circle_inactive)));
                    circle3.setColorFilter((getResources().getColor(R.color.circle_inactive)));


                }
                else if(current==1){
                    viewPager.setCurrentItem(2);
                    circle3.setColorFilter((getResources().getColor(R.color.circle_active)));
                    circle2.setColorFilter((getResources().getColor(R.color.circle_inactive)));
                    circle1.setColorFilter((getResources().getColor(R.color.circle_inactive)));

                }
                else if(current==2){
                    editor.putBoolean("firstLaunch", false);
                    editor.commit();
                    Intent intent = new Intent(IntroSlider.this, Login.class);
                    startActivity(intent);
                    finish();
                }


            }
        });

        viewPager.addOnPageChangeListener(new ViewPager.OnPageChangeListener() {
            @Override
            public void onPageScrolled(int position, float positionOffset, int positionOffsetPixels) {

            }

            @Override
            public void onPageSelected(int position) {

                if(position==0){
                    circle1.setColorFilter((getResources().getColor(R.color.circle_active)));
                    circle2.setColorFilter((getResources().getColor(R.color.circle_inactive)));
                    circle3.setColorFilter((getResources().getColor(R.color.circle_inactive)));


                }
                else if(position==1){
                    circle2.setColorFilter((getResources().getColor(R.color.circle_active)));
                    circle1.setColorFilter((getResources().getColor(R.color.circle_inactive)));
                    circle3.setColorFilter((getResources().getColor(R.color.circle_inactive)));

                }
                else if(position==2){
                    circle3.setColorFilter((getResources().getColor(R.color.circle_active)));
                    circle2.setColorFilter((getResources().getColor(R.color.circle_inactive)));
                    circle1.setColorFilter((getResources().getColor(R.color.circle_inactive)));

                }

            }

            @Override
            public void onPageScrollStateChanged(int state) {

            }
        });




    }
}