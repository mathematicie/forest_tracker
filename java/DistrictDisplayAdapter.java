package com.environment.foresttracker;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;

public class DistrictDisplayAdapter extends RecyclerView.Adapter<DistrictDisplayAdapter.ViewHolder> {

    private final ArrayList<TreeClass> list;

    public DistrictDisplayAdapter(ArrayList<TreeClass> list){
        this.list = list;
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {

        LayoutInflater layoutInflater = LayoutInflater.from(parent.getContext());
        View listItem = layoutInflater.inflate(R.layout.tree_single_data, parent, false);

        return new ViewHolder(listItem);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        String netText = "", treescut, treesplanted;
        int netNumber;

        treescut = list.get(position).treecut;
        treesplanted = list.get(position).treeplanted;
        if (Integer.parseInt(treescut)>Integer.parseInt(treesplanted)) {
            netNumber = Integer.parseInt(treescut)-Integer.parseInt(treesplanted);
            netText ="Loss of "+netNumber+" trees.";
        }
        else if (Integer.parseInt(treescut)<Integer.parseInt(treesplanted)) {
            netNumber = Integer.parseInt(treesplanted)-Integer.parseInt(treescut);
            netText ="Gain of "+netNumber+" trees.";
        }
        else if (Integer.parseInt(treescut)==Integer.parseInt(treesplanted)) {
            netText ="No net gain or loss!";
        }


        holder.District.setText(list.get(position).District);
        holder.treesPlanted.setText("Trees Planted: "+list.get(position).treeplanted);
        holder.treesCut.setText("Trees Cut: "+list.get(position).treecut);
        holder.net.setText(netText);



    }


    @Override
    public int getItemCount() {
        return list.size();
    }


    public static class ViewHolder extends RecyclerView.ViewHolder{

        public TextView District, treesCut, treesPlanted, net;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);

            District = itemView.findViewById(R.id.showSingleDistrict);
            treesCut = itemView.findViewById(R.id.showSingleDataTreesCut);
            treesPlanted = itemView.findViewById(R.id.showSingleDataTreesPlanted);
            net = itemView.findViewById(R.id.showSingleDataNetTrees);

        }
    }

}
