//Package name ---

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;


public class ShowAllUserDataAdapter extends RecyclerView.Adapter<ShowAllUserDataAdapter.ViewHolder> {

        private ArrayList<SaveTreeClass> list;

        public ShowAllUserDataAdapter(ArrayList<SaveTreeClass> list){
            this.list = list;
        }

        @NonNull
        @Override
        public ShowAllUserDataAdapter.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {

            LayoutInflater layoutInflater = LayoutInflater.from(parent.getContext());
            View listItem = layoutInflater.inflate(R.layout.user_single_data, parent, false);
            ShowAllUserDataAdapter.ViewHolder viewHolder = new ShowAllUserDataAdapter.ViewHolder(listItem);

            return viewHolder;
        }

        @Override
        public void onBindViewHolder(@NonNull ShowAllUserDataAdapter.ViewHolder holder, int position) {
            String textStateUT, textDistrict, number = "", cutorplant, textDateTime;

            textStateUT = list.get(position).StateUT;
            textDistrict = list.get(position).District;
            cutorplant = list.get(position).cutOrPlant;
            number = list.get(position).treenumber;
            textDateTime = list.get(position).dateTime;

            String toShow = "";
            if (cutorplant.contains("cut")) toShow="Trees Cut: "+String.valueOf(number);
            else if (cutorplant.contains("plant")) toShow = "Trees Planted: "+String.valueOf(number);

            holder.StateUT.setText(list.get(position).StateUT);
            holder.District.setText(list.get(position).District);
            holder.treeNumber.setText(toShow);
            holder.dateTime.setText(textDateTime);



        }


        @Override
        public int getItemCount() {
            return list.size();
        }


        public static class ViewHolder extends RecyclerView.ViewHolder{

            public TextView StateUT, District, treeNumber, cutorplant, dateTime;

            public ViewHolder(@NonNull View itemView) {
                super(itemView);

                StateUT = itemView.findViewById(R.id.showStateUTSingleData);
                District = itemView.findViewById(R.id.showDistrictSingleData);
                treeNumber = itemView.findViewById(R.id.showTreeNumberSingleData);
                dateTime = itemView.findViewById(R.id.showDateTimeSingleData);

            }
        }


    }
