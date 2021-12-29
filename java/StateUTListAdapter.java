//Package name ---

import android.content.Intent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;
import java.util.HashSet;

public class StateUTListAdapter extends RecyclerView.Adapter<StateUTListAdapter.ViewHolder> {

    private ArrayList<String> list;


    public StateUTListAdapter(ArrayList<String> list){
        this.list = list;
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {

        LayoutInflater layoutInflater = LayoutInflater.from(parent.getContext());
        View listItem = layoutInflater.inflate(R.layout.tree_single_stateut, parent, false);
        ViewHolder viewHolder = new ViewHolder(listItem);

        return viewHolder;
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {




        String textStateUT;
        textStateUT = list.get(position);



        holder.StateUT.setText(textStateUT);


        holder.StateUT.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(holder.itemView.getContext(), Districts.class);
                intent.putExtra("StateUT", textStateUT);
                holder.itemView.getContext().startActivity(intent);
            }
        });


    }


    @Override
    public int getItemCount() {
        return list.size();
    }


    public static class ViewHolder extends RecyclerView.ViewHolder{

        public TextView StateUT;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);

            StateUT = itemView.findViewById(R.id.showSingleStateUT);


        }
    }

}
